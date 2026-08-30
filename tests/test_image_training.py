from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from momo_lm.image_model import TinyCanvasModel
from momo_lm.image_training import (
    ImageManifestError,
    create_reference_manifest,
    load_manifest,
    train_manifest,
)


class ImageTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest_path = self._make_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_manifest(self) -> Path:
        size = 24
        y, x = np.mgrid[0:size, 0:size]
        patterns = {
            "anime": np.stack(
                [
                    0.75 + 0.2 * (x > size // 2),
                    0.35 + 0.35 * (y < size // 2),
                    0.65 + 0.2 * ((x + y) % 6 < 3),
                ],
                axis=-1,
            ),
            "manga": np.repeat(
                (0.15 + 0.75 * (((x // 3 + y // 3) % 2) == 0))[..., None], 3, axis=-1
            ),
            "illustration": np.stack(
                [0.2 + 0.7 * x / size, 0.25 + 0.55 * y / size, 0.7 - 0.45 * x / size],
                axis=-1,
            ),
            "realistic": np.stack(
                [
                    0.2 + 0.35 * y / size,
                    0.35 + 0.4 * (y > size // 2),
                    0.75 - 0.45 * y / size,
                ],
                axis=-1,
            ),
        }
        records = []
        for style, values in patterns.items():
            path = self.root / f"{style}.png"
            Image.fromarray((np.clip(values, 0, 1) * 255).astype(np.uint8)).save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append(
                {
                    "image": path.name,
                    "prompt": f"licensed {style} training scene",
                    "style": style,
                    "negative_prompt": "watermark",
                    "source": "generated-test-fixture",
                    "license": "CC0-1.0",
                    "sha256": digest,
                }
            )
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps({"format_version": 1, "examples": records}), encoding="utf-8"
        )
        return manifest

    def test_manifest_requires_provenance_license_and_matching_hash(self) -> None:
        loaded = load_manifest(self.manifest_path)
        self.assertEqual(len(loaded.examples), 4)
        self.assertEqual({example.style for example in loaded.examples}, set(TinyCanvasModel.STYLES))

        document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        del document["examples"][0]["license"]
        invalid = self.root / "missing-license.json"
        invalid.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ImageManifestError, "fields"):
            load_manifest(invalid)

        document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        document["examples"][0]["sha256"] = "0" * 64
        invalid.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ImageManifestError, "SHA-256 mismatch"):
            load_manifest(invalid)

    def test_reference_manifest_is_reproducible_and_validated(self) -> None:
        first = create_reference_manifest(self.root / "reference-a")
        second = create_reference_manifest(self.root / "reference-b")
        first_manifest = load_manifest(first)
        second_manifest = load_manifest(second)

        self.assertEqual(
            first_manifest.sha256,
            "f486d944d01277acdb30b7de7cc428bf98be890e376de996a163ca2d60e90229",
        )
        self.assertEqual(first_manifest.sha256, second_manifest.sha256)
        self.assertEqual(
            [example.sha256 for example in first_manifest.examples],
            [example.sha256 for example in second_manifest.examples],
        )

    def test_manifest_rejects_path_traversal_and_symlink_escape(self) -> None:
        document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        document["examples"][0]["image"] = "../outside.png"
        invalid = self.root / "traversal.json"
        invalid.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ImageManifestError, "relative"):
            load_manifest(invalid)

        outside = self.root.parent / f"{self.root.name}-outside.png"
        outside.write_bytes((self.root / "anime.png").read_bytes())
        link = self.root / "escape.png"
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError):
            outside.unlink(missing_ok=True)
            return
        try:
            document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            document["examples"][0]["image"] = link.name
            document["examples"][0]["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
            invalid.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ImageManifestError, "escapes"):
                load_manifest(invalid)
        finally:
            outside.unlink(missing_ok=True)

    def test_training_is_deterministic_and_reduces_every_style_loss(self) -> None:
        first = TinyCanvasModel()
        second = TinyCanvasModel()
        first_report = train_manifest(
            first,
            self.manifest_path,
            epochs=12,
            learning_rate=0.08,
            samples_per_image=256,
            seed=123,
        )
        second_report = train_manifest(
            second,
            self.manifest_path,
            epochs=12,
            learning_rate=0.08,
            samples_per_image=256,
            seed=123,
        )

        self.assertLess(first_report.final_loss, first_report.initial_loss)
        self.assertEqual(first_report.steps, 48)
        self.assertEqual(first_report.examples, 48)
        self.assertEqual(first_report.to_dict(), second_report.to_dict())
        json.dumps(first_report.to_dict())
        for style in TinyCanvasModel.STYLES:
            self.assertLess(
                first_report.per_style_final_loss[style],
                first_report.per_style_initial_loss[style],
            )
        for name in first.parameters():
            np.testing.assert_array_equal(first.parameters()[name], second.parameters()[name])

        checkpoint = first.save(self.root / "trained.npz")
        restored = TinyCanvasModel.load(checkpoint)
        self.assertEqual(restored.inspect()["training"]["steps"], 48)
        self.assertEqual(
            restored.inspect()["training"]["manifest_sha256"],
            first_report.manifest_sha256,
        )


if __name__ == "__main__":
    unittest.main()
