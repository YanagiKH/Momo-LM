from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from momo_lm.image_model import ImageCheckpointError, ImageShape, TinyCanvasModel


class TinyCanvasModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_v2_round_trip_preserves_tensors_and_training_metadata(self) -> None:
        model = TinyCanvasModel()
        model.training_steps = 12
        model.training_examples = 12
        model.training_initial_loss = 0.2
        model.training_final_loss = 0.1
        model.training_manifest_sha256 = "a" * 64
        checkpoint = model.save(self.root / "image.npz")

        loaded = TinyCanvasModel.load(checkpoint)

        self.assertEqual(loaded.parameter_count, 3_963)
        self.assertEqual(loaded.inspect()["format_version"], 2)
        self.assertEqual(loaded.inspect()["training"]["steps"], 12)
        for name, value in model.parameters().items():
            np.testing.assert_array_equal(value, loaded.parameters()[name])

    def test_bundled_checkpoint_is_the_trained_v2_artifact(self) -> None:
        checkpoint = (
            Path(__file__).resolve().parents[1]
            / "momo_lm"
            / "assets"
            / "weights"
            / "momo-image-base.npz"
        )
        self.assertEqual(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "92ccb5f37a946bcc478f8cccca3d2b7edb513d51233061c05a40b7d298b16b7c",
        )
        model = TinyCanvasModel.load(checkpoint)
        training = model.inspect()["training"]
        self.assertEqual(training["steps"], 96)
        self.assertEqual(training["examples"], 96)
        self.assertLess(training["final_loss"], training["initial_loss"])

    def test_exact_v1_layout_migrates_with_neutral_style_embeddings(self) -> None:
        model = TinyCanvasModel()
        checkpoint = self.root / "legacy.npz"
        metadata = {"format_version": 1, "shape": asdict(model.shape)}
        np.savez_compressed(
            checkpoint,
            metadata=json.dumps(metadata),
            **{name: getattr(model, name) for name in model.V1_PARAMETER_NAMES},
        )

        migrated = TinyCanvasModel.load(checkpoint)

        self.assertEqual(migrated.inspect()["migrated_from"], 1)
        np.testing.assert_array_equal(
            migrated.style_embedding,
            np.zeros((len(model.STYLES), model.shape.latent_size), dtype=np.float32),
        )
        upgraded = migrated.save(self.root / "upgraded.npz")
        self.assertIsNone(TinyCanvasModel.load(upgraded).inspect()["migrated_from"])

    def test_strict_loader_rejects_tensor_tampering_and_unknown_files(self) -> None:
        checkpoint = TinyCanvasModel().save(self.root / "valid.npz")
        with np.load(checkpoint, allow_pickle=False) as archive:
            contents = {name: np.array(archive[name], copy=True) for name in archive.files}
        contents["w_rgb"][0, 0] += 0.25
        np.savez_compressed(self.root / "tampered.npz", **contents)
        with self.assertRaisesRegex(ImageCheckpointError, "hash mismatch"):
            TinyCanvasModel.load(self.root / "tampered.npz")

        contents["w_rgb"][0, 0] -= 0.25
        contents["surprise"] = np.zeros(1, dtype=np.float32)
        np.savez_compressed(self.root / "unknown.npz", **contents)
        with self.assertRaisesRegex(ImageCheckpointError, "ZIP member"):
            TinyCanvasModel.load(self.root / "unknown.npz")

    def test_strict_loader_rejects_non_finite_v1_tensor(self) -> None:
        model = TinyCanvasModel(ImageShape(seed=17))
        tensors = {name: np.array(getattr(model, name), copy=True) for name in model.V1_PARAMETER_NAMES}
        tensors["b_rgb"][0] = np.nan
        np.savez_compressed(
            self.root / "nan.npz",
            metadata=json.dumps({"format_version": 1, "shape": asdict(model.shape)}),
            **tensors,
        )
        with self.assertRaisesRegex(ImageCheckpointError, "non-finite"):
            TinyCanvasModel.load(self.root / "nan.npz")

    def test_save_refuses_invalid_tensor_without_replacing_checkpoint(self) -> None:
        model = TinyCanvasModel()
        checkpoint = model.save(self.root / "preserved.npz")
        original = checkpoint.read_bytes()
        model.b_rgb[0] = np.inf

        with self.assertRaisesRegex(ImageCheckpointError, "invalid tensor"):
            model.save(checkpoint)

        self.assertEqual(checkpoint.read_bytes(), original)
        TinyCanvasModel.load(checkpoint)

    def test_styles_negative_prompt_and_seed_change_output_deterministically(self) -> None:
        model = TinyCanvasModel()
        outputs = {
            style: model.generate(
                "portrait beneath a pink moon",
                128,
                128,
                42,
                style=style,
                quality="draft",
                tile_size=64,
            ).tobytes()
            for style in model.STYLES
        }
        self.assertEqual(len(set(outputs.values())), len(model.STYLES))
        manga = np.frombuffer(outputs["manga"], dtype=np.uint8).reshape(128, 128, 3)
        np.testing.assert_array_equal(manga[..., 0], manga[..., 1])
        np.testing.assert_array_equal(manga[..., 1], manga[..., 2])
        repeated = model.generate(
            "portrait beneath a pink moon",
            128,
            128,
            42,
            style="anime",
            quality="draft",
            tile_size=64,
        )
        self.assertEqual(outputs["anime"], repeated.tobytes())
        negative = model.generate(
            "portrait beneath a pink moon",
            128,
            128,
            42,
            style="anime",
            negative_prompt="blurred hands",
            quality="draft",
            tile_size=64,
        )
        self.assertNotEqual(outputs["anime"], negative.tobytes())

    def test_tiled_rendering_is_independent_of_tile_boundaries(self) -> None:
        model = TinyCanvasModel()
        small_tiles = model.generate(
            "ink city",
            173,
            137,
            9,
            style="manga",
            quality="high",
            steps=3,
            tile_size=32,
        )
        large_tiles = model.generate(
            "ink city",
            173,
            137,
            9,
            style="manga",
            quality="high",
            steps=3,
            tile_size=128,
        )
        self.assertEqual(small_tiles.tobytes(), large_tiles.tobytes())

    def test_generation_options_are_bounded(self) -> None:
        model = TinyCanvasModel()
        with self.assertRaisesRegex(ValueError, "style"):
            model.generate("x", style="oil")
        with self.assertRaisesRegex(ValueError, "quality"):
            model.generate("x", quality="ultra")
        with self.assertRaisesRegex(ValueError, "steps"):
            model.generate("x", steps=9)
        with self.assertRaisesRegex(ValueError, "tile_size"):
            model.generate("x", tile_size=16)


if __name__ == "__main__":
    unittest.main()
