from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from momo_lm.image_model import TinyCanvasModel  # noqa: E402
from momo_lm.model import NeuralTextModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=26)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    weight_dir = ROOT / "momo_lm" / "assets" / "weights"
    text_path = weight_dir / "momo-text-base.npz"
    image_path = weight_dir / "momo-image-base.npz"
    weight_dir.mkdir(parents=True, exist_ok=True)
    corpus = (ROOT / "momo_lm" / "assets" / "corpus" / "starter.txt").read_text(encoding="utf-8")
    if arguments.force or not text_path.exists():
        model = NeuralTextModel()
        losses = model.train_text(corpus, epochs=arguments.epochs, learning_rate=0.035, batch_size=192)
        model.save(text_path)
        print(f"text: {text_path} loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    if arguments.force or not image_path.exists():
        TinyCanvasModel().save(image_path)
        print(f"image: {image_path}")


if __name__ == "__main__":
    main()
