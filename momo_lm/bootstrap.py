from __future__ import annotations

from pathlib import Path

from .config import MomoConfig
from .image_model import TinyCanvasModel
from .model import NeuralTextModel
from .paths import atomic_copy, package_root


def bundled_weight(name: str) -> Path:
    return package_root() / "assets" / "weights" / name


def initialize_weights(config: MomoConfig, *, force: bool = False) -> dict[str, str]:
    targets = {
        "text": (bundled_weight("momo-text-base.npz"), config.model_path, NeuralTextModel),
        "image": (bundled_weight("momo-image-base.npz"), config.image_model_path, TinyCanvasModel),
    }
    result: dict[str, str] = {}
    for key, (source, target, model_type) in targets.items():
        if target.exists() and not force:
            try:
                if key == "text":
                    NeuralTextModel.load(target, recover=False)
                else:
                    model_type.load(target)
            except (OSError, ValueError):
                pass
            else:
                result[key] = str(target)
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            if key == "text":
                NeuralTextModel.load(source, recover=False)
            else:
                model_type.load(source)
            if source.resolve() != target.resolve():
                atomic_copy(source, target)
                if key == "text":
                    atomic_copy(source, target.with_suffix(target.suffix + ".last-good"))
        else:
            model_type().save(target)
        result[key] = str(target)
    example_source = package_root() / "assets" / "mods" / "example_tools.py.example"
    example_target = config.mods_path / "example_tools.py.example"
    if example_source.exists() and not example_target.exists():
        atomic_copy(example_source, example_target)
    config.save()
    return result
