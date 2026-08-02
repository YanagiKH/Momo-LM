from __future__ import annotations

import shutil
from pathlib import Path

from .config import MomoConfig
from .image_model import TinyCanvasModel
from .model import NeuralTextModel
from .paths import package_root


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
            result[key] = str(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists() and source.resolve() != target.resolve():
            shutil.copy2(source, target)
        else:
            model_type().save(target)
        result[key] = str(target)
    example_source = package_root() / "assets" / "mods" / "example_tools.py.example"
    example_target = config.mods_path / "example_tools.py.example"
    if example_source.exists() and not example_target.exists():
        shutil.copy2(example_source, example_target)
    config.save()
    return result
