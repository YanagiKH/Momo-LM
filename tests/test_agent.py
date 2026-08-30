import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from momo_lm.agent import AgentManager, DeterministicPlanner
from momo_lm.agent_store import AgentStore
from momo_lm.agent_tools import AgentToolbox


class FakeKnowledge:
    def search(self, query: str, limit: int = 4) -> list[SimpleNamespace]:
        return [SimpleNamespace(source="local", content=f"context {query}", score=1.0)][:limit]


class FakeRuntime:
    def __init__(self) -> None:
        self.store = FakeKnowledge()
        self.training_calls: list[str] = []

    @staticmethod
    def status() -> dict[str, object]:
        return {
            "version": "test",
            "compute_backend": {"name": "numpy"},
            "weights": {},
            "image_weights": {},
            "knowledge": {},
            "self_learning": False,
        }

    def train(self, text: str, *, epochs: int, source: str) -> dict[str, object]:
        self.training_calls.append(text)
        return {"chunks": 1, "epochs": epochs, "source": source}


class AgentManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.database_path = root / "agents.db"
        self.runtime = FakeRuntime()
        self.store = AgentStore(self.database_path)
        self.toolbox = AgentToolbox(self.runtime, root / "workspace")
        self.manager = AgentManager(self.store, self.toolbox, recover=False)

    def tearDown(self) -> None:
        self.manager.close()
        self.store.close()
        self.temporary.cleanup()

    def test_planner_is_deterministic_and_profiles_are_supported(self) -> None:
        planner = DeterministicPlanner()
        for profile in ("training", "coding", "workplace", "copilot"):
            first = planner.plan("Prepare a local brief", profile=profile)
            second = planner.plan("Prepare a local brief", profile=profile)
            self.assertEqual(first, second)

    def test_profiles_are_read_only_until_mutating_capability_is_explicit(self) -> None:
        record = self.manager.create("Review the workspace", profile="coding")
        self.assertEqual(record["status"], "completed")
        self.assertNotIn("workspace.write", record["capabilities"])
        with self.assertRaisesRegex(ValueError, "Explicit capability model.train"):
            self.manager.create("train: sample", profile="training")
        with self.assertRaisesRegex(ValueError, "Explicit capability workspace.write"):
            self.manager.create("write: note.txt\nhello", profile="coding")

    def test_mutating_tool_waits_for_exact_one_use_approval(self) -> None:
        record = self.manager.create(
            "train: approved sample",
            profile="training",
            capabilities=["model.train"],
        )
        self.assertEqual(record["status"], "waiting_approval")
        self.assertEqual(self.runtime.training_calls, [])
        approval_id = record["pending_approval"]["id"]
        completed = self.manager.approve(record["id"], approval_id)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(self.runtime.training_calls, ["approved sample"])
        with self.assertRaisesRegex(ValueError, "not waiting"):
            self.manager.approve(record["id"], approval_id)

    def test_cancel_revokes_pending_approval(self) -> None:
        record = self.manager.create(
            "write: result.txt\ncontent",
            profile="coding",
            capabilities=["workspace.write"],
        )
        self.assertEqual(record["status"], "waiting_approval")
        approval_id = record["pending_approval"]["id"]
        cancelled = self.manager.cancel(record["id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertIsNone(cancelled["pending_approval"])
        with self.assertRaisesRegex(ValueError, "not waiting"):
            self.manager.approve(record["id"], approval_id)

    def test_cancel_during_tool_execution_cannot_be_overwritten_by_result(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def slow_draft(_: dict[str, object]) -> dict[str, str]:
            started.set()
            release.wait(timeout=3)
            return {"draft": "finished"}

        self.toolbox._handlers["draft_text"] = slow_draft
        record = self.manager.create("Slow brief", profile="workplace", background=True)
        self.assertTrue(started.wait(timeout=3))
        cancelled = self.manager.cancel(record["id"])
        self.assertEqual(cancelled["status"], "cancelled")
        release.set()
        for _ in range(100):
            final = self.store.get_agent(record["id"])
            if final["status"] == "cancelled":
                break
            time.sleep(0.01)
        self.assertEqual(final["status"], "cancelled")

    def test_budgets_are_enforced_and_validated(self) -> None:
        record = self.manager.create(
            "Draft a brief", profile="workplace", budgets={"max_tool_calls": 0}
        )
        self.assertEqual(record["status"], "failed")
        self.assertIn("max_tool_calls", record["error"])
        planner = DeterministicPlanner()
        for budgets in (
            {"max_steps": 0},
            {"max_steps": 129},
            {"max_input_chars": 255},
            {"max_input_chars": 1_000_001},
            {"max_tool_calls": 1.5},
        ):
            with self.subTest(budgets=budgets), self.assertRaises(ValueError):
                planner.plan("goal", profile="copilot", budgets=budgets)

    def test_results_and_events_redact_credentials(self) -> None:
        record = self.manager.create("Summarize token=top-secret", profile="workplace")
        self.assertEqual(record["status"], "completed")
        serialized = str(record)
        self.assertNotIn("top-secret", serialized)
        events = self.store.events(record["id"])
        self.assertNotIn("top-secret", str(events))

    def test_waiting_approval_survives_restart(self) -> None:
        record = self.manager.create(
            "train: persistent sample",
            profile="training",
            capabilities=["model.train"],
        )
        approval_id = record["pending_approval"]["id"]
        self.manager.close()
        self.store.close()

        self.store = AgentStore(self.database_path)
        self.manager = AgentManager(self.store, self.toolbox, recover=True)
        restored = self.store.get_agent(record["id"])
        self.assertEqual(restored["status"], "waiting_approval")
        completed = self.manager.approve(record["id"], approval_id)
        self.assertEqual(completed["status"], "completed")

    def test_interrupted_read_only_replays_but_mutation_fails_closed(self) -> None:
        readonly = self.store.create_agent(
            goal="inspect",
            profile="copilot",
            capabilities=["runtime.inspect"],
            budgets={"max_steps": 8, "max_tool_calls": 8, "max_input_chars": 12000},
            plan=[{"tool": "inspect_runtime", "arguments": {}, "purpose": "inspect"}],
        )
        self.store.set_running(readonly["id"], 0)

        mutating = self.store.create_agent(
            goal="write",
            profile="coding",
            capabilities=["workspace.write"],
            budgets={"max_steps": 8, "max_tool_calls": 8, "max_input_chars": 12000},
            plan=[
                {
                    "tool": "write_text_file",
                    "arguments": {"path": "result.txt", "content": "value"},
                    "purpose": "write",
                }
            ],
        )
        self.store.set_running(mutating["id"], 0)
        self.manager.recover()

        for _ in range(100):
            read_record = self.store.get_agent(readonly["id"])
            if read_record["status"] == "completed":
                break
            time.sleep(0.01)
        self.assertEqual(read_record["status"], "completed")
        write_record = self.store.get_agent(mutating["id"])
        self.assertEqual(write_record["status"], "failed")
        self.assertIn("not replayed", write_record["error"])
        self.assertFalse((self.toolbox.workspace / "result.txt").exists())


if __name__ == "__main__":
    unittest.main()
