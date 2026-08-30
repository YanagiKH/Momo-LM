import importlib
import tempfile
import unittest
from pathlib import Path

import momo_lm


class PublicApiTests(unittest.TestCase):
    def test_direct_import_and_compatibility_import(self) -> None:
        compatibility = importlib.import_module("MomoLM")
        self.assertIs(compatibility.MomoLM, momo_lm.MomoLM)
        self.assertTrue(callable(momo_lm.load_model))

    def test_embedded_model_chat_training_and_context_manager(self) -> None:
        with tempfile.TemporaryDirectory() as directory, momo_lm.load_model(
            Path(directory)
        ) as model:
            answer = model.chat("你能做什麼？", learn=False)
            self.assertIn("對話", answer)
            result = model.train("Momo Python API code is peach-17.", epochs=1)
            self.assertEqual(result["chunks"], 1)
            self.assertEqual(model.inspect()["version"], "0.3.0")

    def test_embedded_agent_uses_explicit_one_use_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory, momo_lm.load_model(
            Path(directory)
        ) as model:
            waiting = model.create_agent(
                "write: api/result.txt\nAPI content",
                profile="coding",
                capabilities=["workspace.write"],
            )
            self.assertEqual(waiting["status"], "waiting_approval")
            completed = model.approve_agent(
                waiting["id"], waiting["pending_approval"]["id"]
            )
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(
                (Path(directory) / "agent-workspace" / "api" / "result.txt").read_text(
                    encoding="utf-8"
                ),
                "API content",
            )
            self.assertTrue(model.agent_events(waiting["id"]))

    def test_embedded_image_options_reach_v2_generator(self) -> None:
        with tempfile.TemporaryDirectory() as directory, momo_lm.load_model(
            Path(directory)
        ) as model:
            output = Path(directory) / "styled.png"
            generated = model.generate_image(
                "portrait under a moon",
                output,
                width=128,
                height=128,
                seed=7,
                style="manga",
                negative_prompt="watermark",
                quality="draft",
                steps=1,
                tile_size=64,
            )
            self.assertEqual(generated, output)
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
