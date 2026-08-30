import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from momo_lm.model import CheckpointLimits, ModelShape, NeuralTextModel


def small_shape(seed: int = 7) -> ModelShape:
    return ModelShape(
        context_length=8,
        embedding_size=12,
        hidden_size=24,
        attention_heads=3,
        neuron_groups=6,
        seed=seed,
    )


class ModelTests(unittest.TestCase):
    def test_default_v3_architecture_has_expected_size(self) -> None:
        model = NeuralTextModel()
        self.assertEqual(model.parameter_count, 223_835)
        self.assertEqual(model.shape.context_length, 128)
        self.assertEqual(model.shape.attention_heads, 4)
        self.assertEqual(model.shape.neuron_groups, 8)

    def test_training_reduces_loss_and_checkpoint_manifest_is_verified(self) -> None:
        model = NeuralTextModel(small_shape())
        losses = model.train_text(
            "User: hello\nMomo: hello there\n" * 8,
            epochs=6,
            learning_rate=0.004,
            batch_size=32,
            seed=1,
        )
        self.assertLess(losses[-1], losses[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            model.save(path)
            loaded = NeuralTextModel.load(path, recover=False)
            self.assertEqual(loaded.parameter_count, model.parameter_count)
            self.assertEqual(loaded.steps, model.steps)
            np.testing.assert_array_equal(loaded.w_output, model.w_output)
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
            manifest = metadata["tensor_manifest"]
            self.assertEqual(set(manifest), set(model.parameters()))
            self.assertEqual(manifest["w_output"]["shape"], list(model.w_output.shape))
            self.assertEqual(manifest["w_output"]["dtype"], "float32-le")

    def test_checkpoint_bytes_are_reproducible(self) -> None:
        model = NeuralTextModel(small_shape())
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.npz"
            second = Path(directory) / "second.npz"
            model.save(first)
            model.save(second)
            first_hash = hashlib.sha256(first.read_bytes()).digest()
            second_hash = hashlib.sha256(second.read_bytes()).digest()
            self.assertEqual(first_hash, second_hash)

    def test_tampered_tensor_is_rejected_and_last_good_recovers(self) -> None:
        model = NeuralTextModel(small_shape())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            model.save(path)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
            arrays["w_output"][0, 0] += 1.0
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                NeuralTextModel.load(path, recover=False)
            recovered = NeuralTextModel.load(path)
            self.assertEqual(recovered.training_metadata["recovery"]["source"], "last-good")

    def test_compressed_size_limit_is_checked_before_load(self) -> None:
        model = NeuralTextModel(small_shape())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            model.save(path)
            original = NeuralTextModel.CHECKPOINT_LIMITS
            NeuralTextModel.CHECKPOINT_LIMITS = CheckpointLimits(max_compressed_bytes=1)
            try:
                with self.assertRaisesRegex(ValueError, "compressed-size"):
                    NeuralTextModel.load(path, recover=False)
            finally:
                NeuralTextModel.CHECKPOINT_LIMITS = original

    def test_unpacked_limit_and_unexpected_tensor_are_rejected(self) -> None:
        model = NeuralTextModel(small_shape())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            model.save(path)
            original = NeuralTextModel.CHECKPOINT_LIMITS
            NeuralTextModel.CHECKPOINT_LIMITS = CheckpointLimits(
                max_uncompressed_bytes=100
            )
            try:
                with self.assertRaisesRegex(ValueError, "unpacked-size"):
                    NeuralTextModel.load(path, recover=False)
            finally:
                NeuralTextModel.CHECKPOINT_LIMITS = original
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
            arrays["unexpected"] = np.ones(1, dtype=np.float32)
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "missing or unexpected"):
                NeuralTextModel.load(path, recover=False)

    def test_failed_save_does_not_replace_last_valid_checkpoint(self) -> None:
        model = NeuralTextModel(small_shape())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            model.save(path)
            original = path.read_bytes()
            model.w_output[0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "non-finite"):
                model.save(path)
            self.assertEqual(path.read_bytes(), original)

    def test_generation_is_bounded(self) -> None:
        model = NeuralTextModel(small_shape(seed=3))
        output = model.generate("Momo:", max_new_tokens=12, seed=5)
        self.assertLessEqual(len(output.encode("utf-8")), 12)
        with self.assertRaises(ValueError):
            model.generate("Momo:", temperature=float("nan"))
        with self.assertRaises(ValueError):
            model.generate("Momo:", max_new_tokens=4097)

    def test_v1_and_v2_checkpoints_migrate_deterministically(self) -> None:
        shape = small_shape()
        rng = np.random.default_rng(4)
        legacy_shape = {
            "vocab_size": shape.vocab_size,
            "context_length": shape.context_length,
            "embedding_size": shape.embedding_size,
            "hidden_size": shape.hidden_size,
            "seed": shape.seed,
        }
        common = {
            "embedding": rng.normal(size=(shape.vocab_size, shape.embedding_size)).astype(np.float32),
            "w_hidden": rng.normal(
                size=(shape.context_length * shape.embedding_size, shape.hidden_size)
            ).astype(np.float32),
            "b_hidden": rng.normal(size=shape.hidden_size).astype(np.float32),
            "w_output": rng.normal(size=(shape.hidden_size, shape.vocab_size)).astype(np.float32),
            "b_output": rng.normal(size=shape.vocab_size).astype(np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for version in (1, 2):
                metadata = {
                    "format_version": version,
                    "shape": dict(legacy_shape),
                    "steps": 12,
                    "tokens_seen": 34,
                }
                tensors = dict(common)
                if version == 2:
                    metadata["shape"]["neuron_group_size"] = 4
                    tensors.update(
                        {
                            "w_gate": rng.normal(
                                size=(shape.context_length * shape.embedding_size, shape.hidden_size)
                            ).astype(np.float32),
                            "b_gate": rng.normal(size=shape.hidden_size).astype(np.float32),
                            "w_residual": rng.normal(
                                size=(shape.embedding_size, shape.hidden_size)
                            ).astype(np.float32),
                        }
                    )
                path = root / f"legacy-v{version}.npz"
                np.savez_compressed(path, metadata=json.dumps(metadata), **tensors)
                migrated = NeuralTextModel.load(path, recover=False)
                np.testing.assert_array_equal(migrated.embedding, common["embedding"])
                np.testing.assert_array_equal(migrated.w_output, common["w_output"])
                np.testing.assert_allclose(
                    migrated.w_hidden,
                    common["w_hidden"].reshape(
                        shape.context_length, shape.embedding_size, shape.hidden_size
                    ).sum(axis=0),
                )
                self.assertEqual(migrated.steps, 12)
                self.assertEqual(migrated.tokens_seen, 34)
                self.assertEqual(migrated.training_metadata["migration"]["source_format"], version)


if __name__ == "__main__":
    unittest.main()
