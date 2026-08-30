import os
import tempfile
import threading
import unittest
from pathlib import Path

from momo_lm.agent_store import AgentStore, arguments_digest, redact


class AgentStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "agents.db"
        self.store = AgentStore(self.path)

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def create(self) -> dict[str, object]:
        return self.store.create_agent(
            goal="write a note",
            profile="coding",
            capabilities=["workspace.read", "workspace.write"],
            budgets={"max_steps": 8, "max_tool_calls": 8, "max_input_chars": 12000},
            plan=[
                {
                    "tool": "write_text_file",
                    "arguments": {"path": "note.txt", "content": "hello"},
                    "purpose": "test",
                }
            ],
        )

    def test_wal_persistence_and_events_survive_reopen(self) -> None:
        agent = self.create()
        self.assertEqual(self.store.journal_mode, "wal")
        if os.name != "nt":
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
            for suffix in ("-wal", "-shm"):
                auxiliary = Path(f"{self.path}{suffix}")
                if auxiliary.exists():
                    self.assertEqual(auxiliary.stat().st_mode & 0o777, 0o600)
        self.store.close()
        self.store = AgentStore(self.path)
        restored = self.store.get_agent(agent["id"])
        self.assertEqual(restored["goal"], "write a note")
        self.assertEqual(self.store.events(agent["id"])[0]["type"], "created")

    def test_approval_is_exact_and_single_use(self) -> None:
        agent = self.create()
        arguments = {"path": "note.txt", "content": "hello"}
        record = self.store.create_approval(
            agent["id"],
            step_index=0,
            tool="write_text_file",
            arguments=arguments,
            reason="write",
            ttl_seconds=900,
        )
        approval_id = record["pending_approval"]["id"]
        with self.assertRaisesRegex(ValueError, "exact"):
            self.store.consume_approval(
                agent["id"],
                approval_id,
                tool="write_text_file",
                arguments={"path": "different.txt", "content": "hello"},
                step_index=0,
            )
        self.store.consume_approval(
            agent["id"],
            approval_id,
            tool="write_text_file",
            arguments=arguments,
            step_index=0,
        )
        self.assertTrue(
            self.store.has_consumed_approval(
                agent["id"],
                step_index=0,
                tool="write_text_file",
                arguments=arguments,
            )
        )
        with self.assertRaisesRegex(ValueError, "already"):
            self.store.consume_approval(
                agent["id"],
                approval_id,
                tool="write_text_file",
                arguments=arguments,
                step_index=0,
            )

    def test_concurrent_approval_has_exactly_one_winner(self) -> None:
        agent = self.create()
        arguments = {"path": "note.txt", "content": "hello"}
        pending = self.store.create_approval(
            agent["id"],
            step_index=0,
            tool="write_text_file",
            arguments=arguments,
            reason="write",
            ttl_seconds=900,
        )["pending_approval"]
        barrier = threading.Barrier(3)
        results: list[str] = []

        def consume() -> None:
            barrier.wait()
            try:
                self.store.consume_approval(
                    agent["id"],
                    pending["id"],
                    tool="write_text_file",
                    arguments=arguments,
                    step_index=0,
                )
                results.append("ok")
            except ValueError:
                results.append("denied")

        threads = [threading.Thread(target=consume) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=3)
        self.assertCountEqual(results, ["ok", "denied"])

    def test_expired_approval_fails_closed_without_running_action(self) -> None:
        agent = self.create()
        arguments = {"path": "note.txt", "content": "hello"}
        pending = self.store.create_approval(
            agent["id"],
            step_index=0,
            tool="write_text_file",
            arguments=arguments,
            reason="write",
            ttl_seconds=900,
        )["pending_approval"]
        with self.store._lock, self.store._connection:
            self.store._connection.execute(
                "UPDATE approvals SET expires_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00+00:00", pending["id"]),
            )
        with self.assertRaisesRegex(ValueError, "expired"):
            self.store.consume_approval(
                agent["id"],
                pending["id"],
                tool="write_text_file",
                arguments=arguments,
                step_index=0,
            )
        failed = self.store.get_agent(agent["id"])
        self.assertEqual(failed["status"], "failed")
        self.assertIsNone(failed["pending_approval"])
        self.assertIn("approval_expired", {event["type"] for event in self.store.events(agent["id"])})

    def test_secret_redaction_is_recursive(self) -> None:
        value = {
            "token": "abc",
            "nested": [
                "Authorization=secret-value",
                {"password": "pw", "clientSecret": "client-value"},
            ],
        }
        self.assertEqual(
            redact(value),
            {
                "token": "[REDACTED]",
                "nested": [
                    "Authorization=[REDACTED]",
                    {"password": "[REDACTED]", "clientSecret": "[REDACTED]"},
                ],
            },
        )
        self.assertEqual(
            arguments_digest("tool", {"b": 2, "a": 1}),
            arguments_digest("tool", {"a": 1, "b": 2}),
        )


if __name__ == "__main__":
    unittest.main()
