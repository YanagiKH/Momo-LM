import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from momo_lm.agent_tools import AgentToolbox


class FakeKnowledge:
    def search(self, query: str, limit: int = 4) -> list[SimpleNamespace]:
        return [SimpleNamespace(source="local", content=f"known {query}", score=1.0)][:limit]


class FakeRuntime:
    def __init__(self) -> None:
        self.store = FakeKnowledge()
        self.training_calls: list[tuple[str, int, str]] = []

    @staticmethod
    def status() -> dict[str, object]:
        return {
            "version": "test",
            "compute_backend": {"name": "numpy"},
            "weights": {"parameters": 3},
            "image_weights": {"parameters": 4},
            "knowledge": {"documents": 1},
            "self_learning": False,
        }

    def train(self, text: str, *, epochs: int, source: str) -> dict[str, object]:
        self.training_calls.append((text, epochs, source))
        return {"chunks": 1}


class AgentToolboxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "workspace"
        self.runtime = FakeRuntime()
        self.toolbox = AgentToolbox(self.runtime, self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_registry_contains_only_local_tools(self) -> None:
        names = {tool["name"] for tool in self.toolbox.describe()}
        self.assertIn("read_text_file", names)
        self.assertIn("train_text", names)
        self.assertNotIn("browse_web", names)
        self.assertNotIn("shell", names)
        with self.assertRaises(PermissionError):
            self.toolbox.execute("browse_web", {}, capabilities=set())

    def test_workspace_read_write_and_traversal_boundary(self) -> None:
        result = self.toolbox.execute(
            "write_text_file",
            {"path": "notes/momo.txt", "content": "hello"},
            capabilities={"workspace.write"},
        )
        self.assertEqual(result["bytes_written"], 5)
        if os.name != "nt":
            self.assertEqual(self.root.stat().st_mode & 0o777, 0o700)
            self.assertEqual((self.root / "notes").stat().st_mode & 0o777, 0o700)
            self.assertEqual((self.root / "notes" / "momo.txt").stat().st_mode & 0o777, 0o600)
        read = self.toolbox.execute(
            "read_text_file",
            {"path": "notes/momo.txt"},
            capabilities={"workspace.read"},
        )
        self.assertEqual(read["content"], "hello")
        with self.assertRaises(PermissionError):
            self.toolbox.execute(
                "read_text_file", {"path": "../../outside"}, capabilities={"workspace.read"}
            )
        with self.assertRaises(PermissionError):
            self.toolbox.execute(
                "write_text_file",
                {"path": "/tmp/outside", "content": "no"},
                capabilities={"workspace.write"},
            )

    def test_capability_is_required_for_every_tool(self) -> None:
        with self.assertRaises(PermissionError):
            self.toolbox.execute("inspect_runtime", {}, capabilities=set())
        inspected = self.toolbox.execute(
            "inspect_runtime", {}, capabilities={"runtime.inspect"}
        )
        self.assertEqual(inspected["version"], "test")

    def test_train_is_local_and_bounded(self) -> None:
        result = self.toolbox.execute(
            "train_text",
            {"text": "authorized local text", "epochs": 99, "source": "test"},
            capabilities={"model.train"},
        )
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(self.runtime.training_calls, [("authorized local text", 10, "test")])


if __name__ == "__main__":
    unittest.main()
