from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from context_manager import ContextConfig, ContextManager, ModelProfile
from context_manager.artifacts import ArtifactStore
from context_manager.index import RepositoryIndex
from context_manager.manager import ContextOverflowPrevented


class ContextManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.data = Path(self.tmp.name) / "data"
        self.root.mkdir()
        (self.root / "main.py").write_text("class Agent:\n    def solve(self, prompt):\n        return prompt\n\n# important routing decision\n")
        (self.root / "README.md").write_text("Agent context retrieval and context budget notes\n")
        config = ContextConfig(model=ModelProfile(max_context_tokens=1800, output_reserve_tokens=300, tool_reserve_tokens=100, chars_per_token_estimate=4.0), inline_tool_output_chars=80, chunk_lines=20)
        self.manager = ContextManager(self.root, self.data, config, "test-session")
        self.manager.index_repository()

    def tearDown(self):
        self.manager.close()
        self.tmp.cleanup()

    def test_artifact_inspection_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            self.manager.artifacts.inspect("../../../history/example")

    def test_content_addressed_artifact_and_recovery(self):
        result = self.manager.archive_tool_output("error: command failed\n" + "x" * 300, "exec_shell_command", {"command": "pytest"})
        artifact_id = result["metadata"]["artifact_id"]
        self.assertTrue((self.data / "context_data/artifacts/test-session/objects" / artifact_id[:2] / artifact_id[2:]).exists())
        self.assertIn("artifact:", result["content"])
        self.assertIn("command failed", self.manager.retrieve_artifact(artifact_id))

    def test_retrieval_contains_provenance_and_hash(self):
        result = self.manager.assemble([{"role": "user", "content": "How does Agent solve work?"}])
        joined = json.dumps(result.messages)
        self.assertIn("main.py", joined)
        self.assertRegex(joined, r"hash=[0-9a-f]{64}")
        self.assertLessEqual(result.estimated_tokens, result.prompt_limit_tokens)

    def test_stale_hash_refreshes(self):
        first = self.manager.assemble([{"role": "user", "content": "Agent solve"}])
        self.assertIn("return prompt", json.dumps(first.messages))
        (self.root / "main.py").write_text("class Agent:\n    def solve(self, prompt):\n        return prompt + '! changed'\n")
        second = self.manager.assemble([{"role": "user", "content": "Agent solve"}])
        self.assertIn("changed", json.dumps(second.messages))

    def test_budget_prevents_overflow_and_preserves_system(self):
        messages = [{"role": "system", "content": "Never drop this constraint."}]
        messages.extend({"role": "user", "content": "ordinary dialogue " + str(i) + " " + "z" * 500} for i in range(20))
        result = self.manager.assemble(messages)
        self.assertLessEqual(result.estimated_tokens, result.prompt_limit_tokens)
        self.assertIn("Never drop this constraint", json.dumps(result.messages))

    def test_dialogue_order_is_preserved(self):
        messages = [
            {"role": "user", "content": "first request"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "latest request"},
        ]
        result = self.manager.assemble(messages)
        contents = [message["content"] for message in result.messages if message["content"] in {"first request", "first answer", "latest request"}]
        self.assertEqual(contents, ["first request", "first answer", "latest request"])

    def test_full_artifact_is_recoverable_after_prompt_truncation(self):
        content = "A" * 70000 + "MIDDLE_SECRET_VALUE" + "B" * 70000
        result = self.manager.archive_tool_output(content, "exec_shell_command")
        artifact_id = result["metadata"]["artifact_id"]
        recovered = self.manager.retrieve_artifact(artifact_id)
        self.assertIn("MIDDLE_SECRET_VALUE", recovered)
        self.assertEqual(len(recovered), len(content))

    def test_tool_protocol_survives_compaction(self):
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "large output " + "x" * 500},
            {"role": "user", "content": "continue"},
        ]
        result = self.manager.assemble(messages)
        encoded = json.dumps(result.messages)
        self.assertIn("call-1", encoded)
        self.assertIn("artifact:", encoded)

    def test_disabled_mode_preserves_naive_behavior(self):
        self.manager.config.enabled = False
        messages = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}]
        result = self.manager.assemble(messages)
        self.assertEqual(result.messages, messages)

    def test_multithreaded_same_session(self):
        import concurrent.futures
        from context_manager.service import ContextService
        service = ContextService(self.root, self.data / "thread-service")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(service.assemble, "shared", [{"role": "user", "content": f"Agent solve {i}"}]) for i in range(8)]
                results = [future.result() for future in futures]
            self.assertEqual(len(results), 8)
        finally:
            service.close()

    def test_pinning(self):
        messages = [{"role": "user", "content": "must keep this requirement"}]
        identity = self.manager._normalize_message(messages[0], 0, 1)["metadata"]["identity"]
        self.manager.pin(identity)
        result = self.manager.assemble(messages)
        self.assertIn("must keep this requirement", json.dumps(result.messages))


if __name__ == "__main__":
    unittest.main()
