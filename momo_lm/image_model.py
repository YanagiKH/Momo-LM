from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(slots=True)
class ImageShape:
    prompt_features: int = 64
    latent_size: int = 24
    hidden_size: int = 64
    seed: int = 314159


class TinyCanvasModel:
    """Small local coordinate-network image generator for abstract concept art."""

    FORMAT_VERSION = 1

    def __init__(self, shape: ImageShape | None = None) -> None:
        self.shape = shape or ImageShape()
        rng = np.random.default_rng(self.shape.seed)
        p, z, h = self.shape.prompt_features, self.shape.latent_size, self.shape.hidden_size
        self.w_prompt = rng.normal(0, 0.25, (p, z)).astype(np.float32)
        self.b_prompt = rng.normal(0, 0.08, z).astype(np.float32)
        self.w_hidden = rng.normal(0, 0.35, (z + 8, h)).astype(np.float32)
        self.b_hidden = rng.normal(0, 0.1, h).astype(np.float32)
        self.w_rgb = rng.normal(0, 0.3, (h, 3)).astype(np.float32)
        self.b_rgb = rng.normal(0, 0.1, 3).astype(np.float32)

    def _prompt_vector(self, prompt: str, seed: int | None) -> np.ndarray:
        vector = np.zeros(self.shape.prompt_features, dtype=np.float32)
        text = prompt.strip().lower().encode("utf-8")
        for index, value in enumerate(text):
            vector[(value + index * 17) % len(vector)] += 1
        if vector.sum() > 0:
            vector /= vector.sum()
        digest = hashlib.blake2b(text, digest_size=8).digest()
        prompt_seed = int.from_bytes(digest, "little") ^ int(seed or 0)
        noise = np.random.default_rng(prompt_seed).normal(0, 0.08, len(vector))
        return vector + noise.astype(np.float32)

    def generate(self, prompt: str, width: int = 512, height: int = 512, seed: int | None = None) -> Image.Image:
        width = max(128, min(int(width), 1024))
        height = max(128, min(int(height), 1024))
        prompt_vector = self._prompt_vector(prompt, seed)
        latent = np.tanh(prompt_vector @ self.w_prompt + self.b_prompt)
        y, x = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
        coordinates = np.stack(
            [x, y, x * y, x * x, y * y, np.sin(x * 5), np.cos(y * 5), np.sin((x + y) * 4)],
            axis=-1,
        ).astype(np.float32)
        latent_map = np.broadcast_to(latent, (height, width, len(latent)))
        features = np.concatenate([coordinates, latent_map], axis=-1)
        hidden = np.tanh(features @ self.w_hidden + self.b_hidden)
        rgb = 1 / (1 + np.exp(-(hidden @ self.w_rgb + self.b_rgb)))
        grain_rng = np.random.default_rng(int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:8], "little") ^ int(seed or 0))
        rgb = np.clip(rgb + grain_rng.normal(0, 0.012, rgb.shape), 0, 1)
        return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"format_version": self.FORMAT_VERSION, "shape": asdict(self.shape)}
        np.savez_compressed(
            path,
            metadata=json.dumps(metadata),
            w_prompt=self.w_prompt,
            b_prompt=self.b_prompt,
            w_hidden=self.w_hidden,
            b_hidden=self.b_hidden,
            w_rgb=self.w_rgb,
            b_rgb=self.b_rgb,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> TinyCanvasModel:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            model = cls(ImageShape(**metadata["shape"]))
            for name in ("w_prompt", "b_prompt", "w_hidden", "b_hidden", "w_rgb", "b_rgb"):
                setattr(model, name, archive[name].astype(np.float32))
            return model

    def inspect(self) -> dict[str, object]:
        arrays = [self.w_prompt, self.b_prompt, self.w_hidden, self.b_hidden, self.w_rgb, self.b_rgb]
        return {
            "architecture": "prompt-conditioned-coordinate-network",
            "parameters": int(sum(value.size for value in arrays)),
            "shape": asdict(self.shape),
            "output": "128-1024 px RGB abstract concept art",
        }
