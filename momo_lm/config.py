from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paths import default_home, ensure_runtime_dirs


@dataclass(slots=True)
class MomoConfig:
    home: Path
    model_path: Path
    image_model_path: Path
    database_path: Path
    mods_path: Path
    host: str = "127.0.0.1"
    port: int = 7860
    language: str = "zh-TW"
    self_learning: bool = True
    learning_rate: float = 0.025
    temperature: float = 0.78
    top_k: int = 32
    max_new_tokens: int = 180
    max_crawl_pages: int = 8
    request_timeout: float = 10.0

    @classmethod
    def defaults(cls, home: Path | None = None) -> MomoConfig:
        root = (home or default_home()).expanduser().resolve()
        ensure_runtime_dirs(root)
        return cls(
            home=root,
            model_path=root / "weights" / "momo-text-base.npz",
            image_model_path=root / "weights" / "momo-image-base.npz",
            database_path=root / "data" / "momo.db",
            mods_path=root / "mods",
        )

    @classmethod
    def load(cls, path: Path | None = None, home: Path | None = None) -> MomoConfig:
        config = cls.defaults(home)
        target = path or config.home / "config.json"
        if not target.exists():
            return config
        values: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
        path_fields = {"home", "model_path", "image_model_path", "database_path", "mods_path"}
        for key, value in values.items():
            if hasattr(config, key):
                setattr(config, key, Path(value).expanduser() if key in path_fields else value)
        ensure_runtime_dirs(config.home)
        return config

    def save(self, path: Path | None = None) -> Path:
        target = path or self.home / "config.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        values = asdict(self)
        for key in ("home", "model_path", "image_model_path", "database_path", "mods_path"):
            values[key] = str(values[key])
        target.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target
