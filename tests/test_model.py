import json
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

    def test_v1_checkpoint_migrates_to_gated_neuron_groups(self) -> None:
        legacy = NeuralTextModel(ModelShape(context_length=6, embedding_size=8, hidden_size=16))
        metadata = {
            "format_version": 1,
            "shape": {
                "vocab_size": legacy.shape.vocab_size,
                "context_length": legacy.shape.context_length,
                "embedding_size": legacy.shape.embedding_size,
                "hidden_size": legacy.shape.hidden_size,
                "seed": legacy.shape.seed,
            },
            "steps": 12,
            "tokens_seen": 34,
        }
        legacy_names = {"embedding", "w_hidden", "b_hidden", "w_output", "b_output"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-v1.npz"
            np.savez_compressed(
                path,
                metadata=json.dumps(metadata),
                **{name: value for name, value in legacy.parameters().items() if name in legacy_names},
            )
            model = NeuralTextModel.load(path)
        self.assertIn("w_gate", model.parameters())
        self.assertIn("w_residual", model.parameters())
        self.assertEqual(model.steps, 12)
        self.assertEqual(model.tokens_seen, 34)


if __name__ == "__main__":
    unittest.main()
