import unittest

import numpy as np

from momo_lm.model import ModelShape, NeuralTextModel
from momo_lm.training import AdamWConfig, AdamWTrainer


def shape() -> ModelShape:
    return ModelShape(
        context_length=6,
        embedding_size=8,
        hidden_size=16,
        attention_heads=2,
        neuron_groups=4,
        seed=17,
    )


class TrainingTests(unittest.TestCase):
    def test_adamw_is_deterministic(self) -> None:
        text = "User: hello\nMomo: hello\n" * 3
        config = AdamWConfig(epochs=3, learning_rate=0.003, batch_size=11, seed=23)
        first = NeuralTextModel(shape())
        second = NeuralTextModel(shape())
        first_result = AdamWTrainer(first, config).fit(text, validation_text=text)
        second_result = AdamWTrainer(second, config).fit(text, validation_text=text)
        self.assertEqual(first_result.to_dict(), second_result.to_dict())
        for name in first.parameters():
            np.testing.assert_array_equal(first.parameters()[name], second.parameters()[name])

    def test_replay_and_counters_are_recorded(self) -> None:
        model = NeuralTextModel(shape())
        config = AdamWConfig(epochs=2, learning_rate=0.003, batch_size=9, seed=5)
        result = AdamWTrainer(model, config).fit(
            "new fact", replay_texts=["old fact", "older fact"], validation_text="new fact"
        )
        self.assertEqual(result.replay_documents, 2)
        self.assertEqual(model.tokens_seen, result.targets_seen)
        self.assertEqual(model.steps, result.optimizer_steps)
        self.assertFalse(model.training_metadata["validation_is_held_out"])
        self.assertEqual(
            model.training_metadata["last_run"]["targets_seen"], result.targets_seen
        )

    def test_invalid_optimizer_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AdamWConfig(learning_rate=float("nan"))
        with self.assertRaises(ValueError):
            AdamWConfig(batch_size=0)

    def test_validation_split_label_is_explicit(self) -> None:
        model = NeuralTextModel(shape())
        trainer = AdamWTrainer(
            model, AdamWConfig(epochs=1, learning_rate=0.003, batch_size=8, seed=4)
        )
        trainer.fit(
            "training text",
            validation_text="separate validation text",
            validation_is_held_out=True,
        )
        self.assertTrue(model.training_metadata["validation_is_held_out"])
        with self.assertRaises(TypeError):
            trainer.fit("training text", validation_is_held_out=1)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
