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
            self.assertEqual(model.inspect()["version"], "0.2.0")


if __name__ == "__main__":
    unittest.main()
