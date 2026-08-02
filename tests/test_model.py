import tempfile
import unittest
from pathlib import Path

import numpy as np

from momo_lm.model import ModelShape, NeuralTextModel


class ModelTests(unittest.TestCase):
    def test_training_reduces_loss_and_checkpoint_is_safe(self) -> None:
        model = NeuralTextModel(ModelShape(context_length=8, embedding_size=12, hidden_size=24, seed=7))
        losses = model.train_text("User: hello\nMomo: hello there\n" * 12, epochs=10, learning_rate=0.08, seed=1)
        self.assertLess(losses[-1], losses[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            model.save(path)
            loaded = NeuralTextModel.load(path)
            self.assertEqual(loaded.parameter_count, model.parameter_count)
            self.assertEqual(loaded.steps, model.steps)
            np.testing.assert_allclose(loaded.w_output, model.w_output)

    def test_generation_is_bounded(self) -> None:
        model = NeuralTextModel(ModelShape(context_length=6, embedding_size=8, hidden_size=16, seed=3))
        output = model.generate("Momo:", max_new_tokens=12, seed=5)
        self.assertLessEqual(len(output.encode("utf-8")), 12)


if __name__ == "__main__":
    unittest.main()
