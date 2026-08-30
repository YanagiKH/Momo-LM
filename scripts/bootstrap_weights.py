from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MOMO_BACKEND", "numpy")

from momo_lm.evaluation import evaluate_text  # noqa: E402
from momo_lm.image_model import TinyCanvasModel  # noqa: E402
from momo_lm.image_training import create_reference_manifest, train_manifest  # noqa: E402
from momo_lm.model import NeuralTextModel  # noqa: E402
from momo_lm.training import AdamWConfig, AdamWTrainer  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproducibly train and write the bundled Momo-LM checkpoints."
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=144)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-image", action="store_true")
    arguments = parser.parse_args()

    weight_dir = ROOT / "momo_lm" / "assets" / "weights"
    text_path = weight_dir / "momo-text-base.npz"
    image_path = weight_dir / "momo-image-base.npz"
    corpus_path = ROOT / "momo_lm" / "assets" / "corpus" / "starter.txt"
    weight_dir.mkdir(parents=True, exist_ok=True)
    corpus = corpus_path.read_text(encoding="utf-8")

    if arguments.force or not text_path.exists():
        model = NeuralTextModel()
        baseline = evaluate_text(model, corpus, batch_size=arguments.batch_size)
        trainer = AdamWTrainer(
            model,
            AdamWConfig(
                epochs=arguments.epochs,
                learning_rate=arguments.learning_rate,
                batch_size=arguments.batch_size,
                seed=arguments.seed,
            ),
        )
        result = trainer.fit(corpus, validation_text=corpus)
        final = result.validation or evaluate_text(
            model, corpus, batch_size=arguments.batch_size
        )
        model.training_metadata["provenance"] = {
            "corpus": "momo_lm/assets/corpus/starter.txt",
            "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
            "corpus_utf8_bytes": len(corpus.encode("utf-8")),
            "validation_protocol": "starter-corpus-overlap-not-held-out",
            "baseline": baseline.to_dict(),
            "final": final.to_dict(),
        }
        model.save(text_path)
        text_path.with_suffix(text_path.suffix + ".last-good").unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "text": str(text_path),
                    "sha256": file_sha256(text_path),
                    "parameters": model.parameter_count,
                    "steps": model.steps,
                    "tokens_seen": model.tokens_seen,
                    "loss": [result.epoch_losses[0], result.epoch_losses[-1]],
                    "baseline": baseline.to_dict(),
                    "final": final.to_dict(),
                    "validation_is_held_out": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    else:
        model = NeuralTextModel.load(text_path, recover=False)
        print(
            json.dumps(
                {
                    "text": str(text_path),
                    "sha256": file_sha256(text_path),
                    "parameters": model.parameter_count,
                    "steps": model.steps,
                    "tokens_seen": model.tokens_seen,
                    "status": "kept-existing",
                },
                sort_keys=True,
            )
        )

    if not arguments.skip_image:
        image_report = None
        if arguments.force or not image_path.exists():
            image_model = TinyCanvasModel()
            with tempfile.TemporaryDirectory(prefix="momo-image-reference-") as directory:
                manifest = create_reference_manifest(Path(directory))
                image_report = train_manifest(
                    image_model,
                    manifest,
                    epochs=24,
                    learning_rate=0.5,
                    samples_per_image=512,
                    seed=20260830,
                )
            image_model.save(image_path)
        print(
            json.dumps(
                {
                    "image": str(image_path),
                    "sha256": file_sha256(image_path),
                    "training": image_report.to_dict() if image_report else None,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
