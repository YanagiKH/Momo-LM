import unittest

from momo_lm.evaluation import compare_text, evaluate_text
from momo_lm.model import ModelShape, NeuralTextModel


class EvaluationTests(unittest.TestCase):
    def test_metrics_are_finite_and_do_not_mutate_counters(self) -> None:
        model = NeuralTextModel(
            ModelShape(
                context_length=6,
                embedding_size=8,
                hidden_size=16,
                attention_heads=2,
                neuron_groups=4,
                seed=9,
            )
        )
        before = (model.steps, model.tokens_seen)
        metrics = evaluate_text(model, "User: hi\nMomo: hello\n", batch_size=7)
        self.assertGreater(metrics.negative_log_likelihood, 0)
        self.assertGreaterEqual(metrics.top1_accuracy, 0)
        self.assertLessEqual(metrics.top1_accuracy, 1)
        self.assertEqual((model.steps, model.tokens_seen), before)

    def test_comparison_reports_nll_improvement(self) -> None:
        shape = ModelShape(
            context_length=6,
            embedding_size=8,
            hidden_size=16,
            attention_heads=2,
            neuron_groups=4,
            seed=11,
        )
        baseline = NeuralTextModel(shape)
        candidate = NeuralTextModel(shape)
        candidate.train_text(
            "hello hello hello", epochs=4, learning_rate=0.004, batch_size=8, seed=3
        )
        comparison = compare_text(baseline, candidate, "hello hello hello", batch_size=8)
        self.assertTrue(comparison["nll_improved"])


if __name__ == "__main__":
    unittest.main()
