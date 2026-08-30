import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from momo_lm.cli import main


class CliTests(unittest.TestCase):
    def call(self, arguments: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_agent_run_list_status_events_and_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = ["--home", directory, "agent"]
            code, output, _ = self.call(
                [*prefix, "run", "Prepare a local brief", "--profile", "workplace"]
            )
            self.assertEqual(code, 0)
            created = json.loads(output)
            self.assertEqual(created["status"], "completed")
            agent_id = created["id"]

            code, output, _ = self.call([*prefix, "status", agent_id])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output)["agent"]["id"], agent_id)
            code, output, _ = self.call([*prefix, "events", agent_id])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(output)["events"])
            code, output, _ = self.call([*prefix, "list"])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(output)["agents"])

    def test_agent_write_requires_then_consumes_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = ["--home", directory, "agent"]
            code, output, _ = self.call(
                [
                    *prefix,
                    "run",
                    "write: notes/result.txt\napproved content",
                    "--profile",
                    "coding",
                    "--capability",
                    "workspace.write",
                ]
            )
            self.assertEqual(code, 0)
            waiting = json.loads(output)
            self.assertEqual(waiting["status"], "waiting_approval")
            code, output, _ = self.call(
                [
                    *prefix,
                    "approve",
                    waiting["id"],
                    waiting["pending_approval"]["id"],
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output)["agent"]["status"], "completed")
            self.assertEqual(
                (Path(directory) / "agent-workspace" / "notes" / "result.txt").read_text(
                    encoding="utf-8"
                ),
                "approved content",
            )

    def test_agent_rejects_unapproved_capability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code, _, error = self.call(
                [
                    "--home",
                    directory,
                    "agent",
                    "run",
                    "goal",
                    "--profile",
                    "workplace",
                    "--capability",
                    "model.train",
                ]
            )
            self.assertEqual(code, 2)
            self.assertIn("not allowed", error)

    def test_image_command_exposes_v2_style_quality_and_negative_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cli-image.png"
            code, printed, error = self.call(
                [
                    "--home",
                    directory,
                    "image",
                    "portrait",
                    "--output",
                    str(output),
                    "--width",
                    "128",
                    "--height",
                    "128",
                    "--seed",
                    "11",
                    "--style",
                    "realistic",
                    "--negative-prompt",
                    "watermark",
                    "--quality",
                    "draft",
                    "--steps",
                    "1",
                    "--tile-size",
                    "64",
                ]
            )
            self.assertEqual(error, "")
            self.assertEqual(code, 0)
            self.assertEqual(Path(printed.strip()), output.resolve())
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
