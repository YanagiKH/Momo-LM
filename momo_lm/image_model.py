from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image


class ImageCheckpointError(ValueError):
    """Raised when a TinyCanvas checkpoint is malformed or unsafe to load."""


@dataclass(slots=True)
class ImageShape:
    prompt_features: int = 64
    latent_size: int = 24
    hidden_size: int = 64
    seed: int = 314159


class TinyCanvasModel:
    """Small, trainable prompt-conditioned coordinate image network.

    TinyCanvas is deliberately compact and runs entirely in NumPy. It is useful
    for inspecting the image-training path and making stylised local images; it
    is not a diffusion model. Rendering is tiled, so output memory is bounded by
    the selected tile size rather than the full coordinate activation tensor.
    """

    FORMAT_VERSION = 2
    SUPPORTED_FORMATS: ClassVar[frozenset[int]] = frozenset({1, 2})
    STYLES: ClassVar[tuple[str, ...]] = ("anime", "manga", "illustration", "realistic")
    QUALITY_STEPS: ClassVar[dict[str, int]] = {"draft": 1, "standard": 2, "high": 4}
    PARAMETER_NAMES: ClassVar[tuple[str, ...]] = (
        "w_prompt",
        "b_prompt",
        "style_embedding",
        "w_hidden",
        "b_hidden",
        "w_rgb",
        "b_rgb",
    )
    V1_PARAMETER_NAMES: ClassVar[tuple[str, ...]] = (
        "w_prompt",
        "b_prompt",
        "w_hidden",
        "b_hidden",
        "w_rgb",
        "b_rgb",
    )
    MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
    MAX_UNPACKED_BYTES = 128 * 1024 * 1024
    MAX_METADATA_BYTES = 128 * 1024
    MAX_PROMPT_BYTES = 16 * 1024
    MAX_OUTPUT_SIZE = 2048
    MAX_STEPS = 8

    def __init__(self, shape: ImageShape | None = None) -> None:
        self.shape = shape or ImageShape()
        self._validate_shape(self.shape)
        rng = np.random.default_rng(self.shape.seed)
        p, z, h = self.shape.prompt_features, self.shape.latent_size, self.shape.hidden_size
        self.w_prompt = rng.normal(0, 0.25, (p, z)).astype(np.float32)
        self.b_prompt = rng.normal(0, 0.08, z).astype(np.float32)
        self.style_embedding = rng.normal(0, 0.035, (len(self.STYLES), z)).astype(np.float32)
        self.w_hidden = rng.normal(0, 0.35, (z + 8, h)).astype(np.float32)
        self.b_hidden = rng.normal(0, 0.1, h).astype(np.float32)
        self.w_rgb = rng.normal(0, 0.3, (h, 3)).astype(np.float32)
        self.b_rgb = rng.normal(0, 0.1, 3).astype(np.float32)
        self.training_steps = 0
        self.training_examples = 0
        self.training_initial_loss: float | None = None
        self.training_final_loss: float | None = None
        self.training_manifest_sha256: str | None = None
        self.migrated_from: int | None = None

    @property
    def parameter_count(self) -> int:
        return int(sum(value.size for value in self.parameters().values()))

    def parameters(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in self.PARAMETER_NAMES}

    @classmethod
    def _validate_shape(cls, shape: ImageShape) -> None:
        limits = {
            "prompt_features": (8, 4096),
            "latent_size": (4, 2048),
            "hidden_size": (8, 4096),
            "seed": (0, (1 << 63) - 1),
        }
        for name, (minimum, maximum) in limits.items():
            value = getattr(shape, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ImageCheckpointError(
                    f"shape.{name} must be an integer in [{minimum}, {maximum}]"
                )

    @classmethod
    def _validate_prompt(cls, prompt: str, name: str = "prompt") -> str:
        if not isinstance(prompt, str):
            raise TypeError(f"{name} must be a string")
        encoded = prompt.encode("utf-8")
        if len(encoded) > cls.MAX_PROMPT_BYTES:
            raise ValueError(f"{name} exceeds {cls.MAX_PROMPT_BYTES} UTF-8 bytes")
        return prompt.strip()

    @classmethod
    def _style_index(cls, style: str) -> int:
        if not isinstance(style, str) or style not in cls.STYLES:
            choices = ", ".join(cls.STYLES)
            raise ValueError(f"style must be one of: {choices}")
        return cls.STYLES.index(style)

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        positive = value >= 0
        result = np.empty_like(value, dtype=np.float32)
        result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
        exponential = np.exp(value[~positive])
        result[~positive] = exponential / (1.0 + exponential)
        return result

    def _prompt_vector(self, prompt: str, seed: int | None) -> np.ndarray:
        prompt = self._validate_prompt(prompt)
        vector = np.zeros(self.shape.prompt_features, dtype=np.float32)
        text = prompt.lower().encode("utf-8")
        for index, value in enumerate(text):
            vector[(value + index * 17) % len(vector)] += 1.0
            vector[(value * 7 + index * 29) % len(vector)] += 0.35
        total = float(vector.sum())
        if total > 0:
            vector /= total
        digest = hashlib.blake2b(text, digest_size=8, person=b"MomoImg2").digest()
        numeric_seed = int(seed or 0) & ((1 << 64) - 1)
        prompt_seed = int.from_bytes(digest, "little") ^ numeric_seed
        noise = np.random.default_rng(prompt_seed).normal(0, 0.06, len(vector))
        return vector + noise.astype(np.float32)

    def _latent(
        self,
        prompt: str,
        *,
        style: str,
        negative_prompt: str = "",
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        style_index = self._style_index(style)
        positive = self._prompt_vector(prompt, seed)
        negative_prompt = self._validate_prompt(negative_prompt, "negative_prompt")
        if negative_prompt:
            negative_seed = (int(seed or 0) & ((1 << 64) - 1)) ^ 0xA5A5A5A5
            negative = self._prompt_vector(negative_prompt, negative_seed)
            conditioned = positive - 0.65 * negative
        else:
            conditioned = positive
        preactivation = conditioned @ self.w_prompt + self.b_prompt
        preactivation += self.style_embedding[style_index]
        return np.tanh(preactivation).astype(np.float32), conditioned

    @staticmethod
    def coordinate_features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the eight coordinate features used by rendering and training."""

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if x.shape != y.shape:
            raise ValueError("x and y coordinates must have the same shape")
        return np.stack(
            [x, y, x * y, x * x, y * y, np.sin(x * 5), np.cos(y * 5), np.sin((x + y) * 4)],
            axis=-1,
        ).astype(np.float32)

    def _forward_features(
        self, coordinates: np.ndarray, latent: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coordinates = np.asarray(coordinates, dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] != 8:
            raise ValueError("coordinates must have shape (samples, 8)")
        latent_map = np.broadcast_to(latent, (len(coordinates), len(latent)))
        features = np.concatenate([coordinates, latent_map], axis=1)
        # ``optimize=False`` keeps each row's reduction order independent of the
        # tile batch shape. The same seed therefore produces byte-identical
        # pixels even when callers choose a different tile size.
        hidden_projection = np.einsum(
            "ni,ih->nh", features, self.w_hidden, optimize=False
        )
        hidden = np.tanh(hidden_projection + self.b_hidden).astype(np.float32)
        rgb_projection = np.einsum("nh,hc->nc", hidden, self.w_rgb, optimize=False)
        rgb = self._sigmoid(rgb_projection + self.b_rgb)
        return rgb, hidden, features

    @staticmethod
    def _pixel_noise(
        x_indices: np.ndarray, y_indices: np.ndarray, seed: int, amplitude: float
    ) -> np.ndarray:
        phase = (
            (x_indices.astype(np.float64) + 1.0) * 12.9898
            + (y_indices.astype(np.float64) + 1.0) * 78.233
            + float(seed % 1_000_003) * 0.017
        )
        channels = []
        for offset in (0.0, 19.19, 47.77):
            value = np.sin(phase + offset) * 43758.5453123
            channels.append((value - np.floor(value) - 0.5) * 2.0 * amplitude)
        return np.stack(channels, axis=-1).astype(np.float32)

    @staticmethod
    def _apply_style(
        rgb: np.ndarray,
        style: str,
        x_indices: np.ndarray,
        y_indices: np.ndarray,
    ) -> np.ndarray:
        """Apply a small deterministic style prior after neural rendering."""

        if style == "anime":
            mean = rgb.mean(axis=-1, keepdims=True)
            saturated = mean + 1.22 * (rgb - mean)
            return np.round(np.clip(saturated, 0.0, 1.0) * 7.0) / 7.0
        if style == "manga":
            grey = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
            contrasted = np.clip((grey - 0.5) * 1.45 + 0.5, 0.0, 1.0)
            posterised = np.round(contrasted * 5.0) / 5.0
            screen = ((x_indices // 2 + y_indices // 2) % 2).astype(np.float32)
            toned = np.clip(posterised + (screen - 0.5) * 0.035, 0.0, 1.0)
            return np.repeat(toned[..., None], 3, axis=-1)
        if style == "illustration":
            warm = rgb * np.asarray([1.05, 1.0, 0.96], dtype=np.float32)
            mean = warm.mean(axis=-1, keepdims=True)
            return np.clip(mean + 1.08 * (warm - mean), 0.0, 1.0)
        mean = rgb.mean(axis=-1, keepdims=True)
        natural = mean + 0.92 * (rgb - mean)
        return np.clip((natural - 0.5) * 1.08 + 0.5, 0.0, 1.0)

    @staticmethod
    def _sample_offset(step: int, steps: int) -> tuple[float, float]:
        if steps == 1:
            return 0.0, 0.0
        sequence = (
            (-0.25, -0.25),
            (0.25, 0.25),
            (-0.25, 0.25),
            (0.25, -0.25),
            (0.0, -0.33),
            (0.0, 0.33),
            (-0.33, 0.0),
            (0.33, 0.0),
        )
        return sequence[step]

    @classmethod
    def _generation_options(
        cls, quality: str, steps: int | None, tile_size: int
    ) -> tuple[int, int, float]:
        if not isinstance(quality, str) or quality not in cls.QUALITY_STEPS:
            choices = ", ".join(cls.QUALITY_STEPS)
            raise ValueError(f"quality must be one of: {choices}")
        if steps is None:
            resolved_steps = cls.QUALITY_STEPS[quality]
        elif type(steps) is not int or not 1 <= steps <= cls.MAX_STEPS:
            raise ValueError(f"steps must be an integer in [1, {cls.MAX_STEPS}]")
        else:
            resolved_steps = steps
        if type(tile_size) is not int or not 32 <= tile_size <= 512:
            raise ValueError("tile_size must be an integer in [32, 512]")
        grain = {"draft": 0.014, "standard": 0.008, "high": 0.003}[quality]
        return resolved_steps, tile_size, grain

    def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        seed: int | None = None,
        *,
        style: str = "illustration",
        negative_prompt: str = "",
        quality: str = "standard",
        steps: int | None = None,
        tile_size: int = 128,
    ) -> Image.Image:
        """Generate an RGB image with deterministic tiled supersampling."""

        if type(width) is not int or type(height) is not int:
            raise TypeError("width and height must be integers")
        width = max(128, min(width, self.MAX_OUTPUT_SIZE))
        height = max(128, min(height, self.MAX_OUTPUT_SIZE))
        if seed is not None and type(seed) is not int:
            raise TypeError("seed must be an integer or None")
        resolved_steps, tile_size, grain = self._generation_options(quality, steps, tile_size)
        latent, _ = self._latent(
            prompt,
            style=style,
            negative_prompt=negative_prompt,
            seed=seed,
        )
        output = np.empty((height, width, 3), dtype=np.uint8)
        numeric_seed = int(seed or 0)
        for top in range(0, height, tile_size):
            bottom = min(top + tile_size, height)
            y_pixels = np.arange(top, bottom, dtype=np.float32)
            for left in range(0, width, tile_size):
                right = min(left + tile_size, width)
                x_pixels = np.arange(left, right, dtype=np.float32)
                accumulated = np.zeros((bottom - top, right - left, 3), dtype=np.float32)
                for step in range(resolved_steps):
                    offset_x, offset_y = self._sample_offset(step, resolved_steps)
                    x = (2.0 * (x_pixels + 0.5 + offset_x) / width - 1.0).astype(np.float32)
                    y = (2.0 * (y_pixels + 0.5 + offset_y) / height - 1.0).astype(np.float32)
                    grid_x, grid_y = np.meshgrid(x, y)
                    coordinates = self.coordinate_features(grid_x.reshape(-1), grid_y.reshape(-1))
                    rgb, _, _ = self._forward_features(coordinates, latent)
                    accumulated += rgb.reshape(bottom - top, right - left, 3)
                accumulated /= float(resolved_steps)
                grid_x_index, grid_y_index = np.meshgrid(
                    np.arange(left, right, dtype=np.int64),
                    np.arange(top, bottom, dtype=np.int64),
                )
                accumulated = self._apply_style(
                    accumulated
                    + self._pixel_noise(grid_x_index, grid_y_index, numeric_seed, grain),
                    style,
                    grid_x_index,
                    grid_y_index,
                )
                output[top:bottom, left:right] = (
                    np.clip(accumulated, 0.0, 1.0) * 255.0
                ).astype(np.uint8)
        return Image.fromarray(output)

    def train_batch(
        self,
        prompt: str,
        style: str,
        coordinates: np.ndarray,
        targets: np.ndarray,
        *,
        negative_prompt: str = "",
        learning_rate: float = 0.02,
        gradient_clip: float = 1.0,
    ) -> float:
        """Apply one deterministic manual-backpropagation update."""

        if not math.isfinite(learning_rate) or not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be finite and in (0, 1]")
        if not math.isfinite(gradient_clip) or gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be finite and positive")
        style_index = self._style_index(style)
        coordinates = np.asarray(coordinates, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] != 8:
            raise ValueError("coordinates must have shape (samples, 8)")
        if targets.shape != (len(coordinates), 3):
            raise ValueError("targets must have shape (samples, 3)")
        if len(coordinates) == 0:
            raise ValueError("training batch cannot be empty")
        if not np.isfinite(coordinates).all() or not np.isfinite(targets).all():
            raise ValueError("training data must contain only finite values")
        if float(targets.min()) < 0.0 or float(targets.max()) > 1.0:
            raise ValueError("targets must be normalised to [0, 1]")

        latent, conditioned = self._latent(
            prompt,
            style=style,
            negative_prompt=negative_prompt,
            seed=0,
        )
        rgb, hidden, features = self._forward_features(coordinates, latent)
        difference = rgb - targets
        loss = float(np.mean(difference * difference))

        grad_rgb = (2.0 / difference.size) * difference
        grad_logits = grad_rgb * rgb * (1.0 - rgb)
        grad_w_rgb = hidden.T @ grad_logits
        grad_b_rgb = grad_logits.sum(axis=0)
        grad_hidden = grad_logits @ self.w_rgb.T
        grad_hidden_pre = grad_hidden * (1.0 - hidden * hidden)
        grad_w_hidden = features.T @ grad_hidden_pre
        grad_b_hidden = grad_hidden_pre.sum(axis=0)
        grad_features = grad_hidden_pre @ self.w_hidden.T
        grad_latent = grad_features[:, 8:].sum(axis=0)
        grad_latent_pre = grad_latent * (1.0 - latent * latent)
        grad_w_prompt = np.outer(conditioned, grad_latent_pre)
        grad_b_prompt = grad_latent_pre
        grad_style = np.zeros_like(self.style_embedding)
        grad_style[style_index] = grad_latent_pre

        gradients = {
            "w_prompt": grad_w_prompt,
            "b_prompt": grad_b_prompt,
            "style_embedding": grad_style,
            "w_hidden": grad_w_hidden,
            "b_hidden": grad_b_hidden,
            "w_rgb": grad_w_rgb,
            "b_rgb": grad_b_rgb,
        }
        if not math.isfinite(loss) or any(
            not np.isfinite(gradient).all() for gradient in gradients.values()
        ):
            raise FloatingPointError("image training produced non-finite gradients")
        squared_norm = sum(
            float(np.sum(gradient.astype(np.float64) ** 2)) for gradient in gradients.values()
        )
        scale = min(1.0, gradient_clip / (math.sqrt(squared_norm) + 1e-12))
        for name, gradient in gradients.items():
            parameter = getattr(self, name)
            parameter -= np.asarray(learning_rate * scale * gradient, dtype=np.float32)
        self.training_steps += 1
        self.training_examples += 1
        return loss

    def loss(
        self,
        prompt: str,
        style: str,
        coordinates: np.ndarray,
        targets: np.ndarray,
        *,
        negative_prompt: str = "",
    ) -> float:
        """Measure mean squared RGB error without changing model state."""

        latent, _ = self._latent(
            prompt,
            style=style,
            negative_prompt=negative_prompt,
            seed=0,
        )
        rgb, _, _ = self._forward_features(coordinates, latent)
        target_array = np.asarray(targets, dtype=np.float32)
        if target_array.shape != rgb.shape or not np.isfinite(target_array).all():
            raise ValueError("targets must be finite and match the rendered RGB shape")
        return float(np.mean((rgb - target_array) ** 2))

    @staticmethod
    def _tensor_digest(value: np.ndarray) -> str:
        contiguous = np.ascontiguousarray(value.astype("<f4", copy=False))
        return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()

    def _training_metadata(self) -> dict[str, object]:
        return {
            "steps": self.training_steps,
            "examples": self.training_examples,
            "initial_loss": self.training_initial_loss,
            "final_loss": self.training_final_loss,
            "manifest_sha256": self.training_manifest_sha256,
        }

    def save(self, path: Path) -> Path:
        """Atomically save a strict v2 checkpoint with per-tensor hashes."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = self.parameters()
        expected_shapes = self._expected_shapes(self.shape)
        for name, value in tensors.items():
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype("float32")
                or value.shape != expected_shapes[name]
                or not np.isfinite(value).all()
            ):
                raise ImageCheckpointError(f"cannot save invalid tensor {name}")
        training_metadata = self._validate_training_metadata(self._training_metadata())
        tensor_manifest = {
            name: {
                "shape": list(value.shape),
                "dtype": "float32",
                "nbytes": value.nbytes,
                "sha256": self._tensor_digest(value),
            }
            for name, value in tensors.items()
        }
        metadata = {
            "format_version": self.FORMAT_VERSION,
            "architecture": "tinycanvas-coordinate-v2",
            "shape": asdict(self.shape),
            "styles": list(self.STYLES),
            "training": training_metadata,
            "tensors": tensor_manifest,
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(
                    handle,
                    metadata=np.asarray(
                        json.dumps(
                            metadata,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                    **tensors,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name)
            raise
        self.migrated_from = None
        return path

    @classmethod
    def _parse_metadata(cls, raw: np.ndarray) -> dict[str, object]:
        if raw.shape != () or raw.dtype.kind not in {"U", "S"}:
            raise ImageCheckpointError("metadata must be a scalar UTF-8 JSON string")
        value = raw.item()
        if isinstance(value, bytes):
            try:
                encoded = value
                text = value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ImageCheckpointError("metadata is not valid UTF-8") from error
        else:
            text = str(value)
            encoded = text.encode("utf-8")
        if len(encoded) > cls.MAX_METADATA_BYTES:
            raise ImageCheckpointError("checkpoint metadata is too large")
        try:
            metadata = json.loads(text)
        except (json.JSONDecodeError, TypeError) as error:
            raise ImageCheckpointError("checkpoint metadata is not valid JSON") from error
        if not isinstance(metadata, dict):
            raise ImageCheckpointError("checkpoint metadata must be an object")
        return metadata

    @classmethod
    def _shape_from_metadata(cls, metadata: dict[str, object]) -> ImageShape:
        raw_shape = metadata.get("shape")
        expected = {"prompt_features", "latent_size", "hidden_size", "seed"}
        if not isinstance(raw_shape, dict) or set(raw_shape) != expected:
            raise ImageCheckpointError("checkpoint shape fields are invalid")
        try:
            shape = ImageShape(**raw_shape)
        except TypeError as error:
            raise ImageCheckpointError("checkpoint shape is invalid") from error
        cls._validate_shape(shape)
        return shape

    @classmethod
    def _expected_shapes(cls, shape: ImageShape) -> dict[str, tuple[int, ...]]:
        return {
            "w_prompt": (shape.prompt_features, shape.latent_size),
            "b_prompt": (shape.latent_size,),
            "style_embedding": (len(cls.STYLES), shape.latent_size),
            "w_hidden": (shape.latent_size + 8, shape.hidden_size),
            "b_hidden": (shape.hidden_size,),
            "w_rgb": (shape.hidden_size, 3),
            "b_rgb": (3,),
        }

    @classmethod
    def _validate_training_metadata(cls, value: object) -> dict[str, object]:
        fields = {"steps", "examples", "initial_loss", "final_loss", "manifest_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise ImageCheckpointError("training metadata fields are invalid")
        for name in ("steps", "examples"):
            item = value[name]
            if type(item) is not int or not 0 <= item <= (1 << 63) - 1:
                raise ImageCheckpointError(f"training.{name} is invalid")
        for name in ("initial_loss", "final_loss"):
            item = value[name]
            if item is not None and (
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not math.isfinite(float(item))
                or float(item) < 0.0
            ):
                raise ImageCheckpointError(f"training.{name} is invalid")
        digest = value["manifest_sha256"]
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ImageCheckpointError("training.manifest_sha256 is invalid")
        return value

    @classmethod
    def _preflight_checkpoint(cls, path: Path) -> None:
        """Bound ZIP members before NumPy allocates their uncompressed arrays."""

        try:
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                names = [entry.filename for entry in entries]
                allowed = {"metadata.npy", *(f"{name}.npy" for name in cls.PARAMETER_NAMES)}
                if not entries or len(entries) > len(cls.PARAMETER_NAMES) + 1:
                    raise ImageCheckpointError("checkpoint ZIP member count is invalid")
                if len(names) != len(set(names)) or not set(names) <= allowed:
                    raise ImageCheckpointError("checkpoint ZIP members are invalid")
                total = 0
                for entry in entries:
                    if entry.flag_bits & 0x1 or entry.is_dir():
                        raise ImageCheckpointError("checkpoint ZIP member is invalid")
                    if entry.file_size < 0 or entry.compress_size < 0:
                        raise ImageCheckpointError("checkpoint ZIP member size is invalid")
                    total += entry.file_size
                    if total > cls.MAX_UNPACKED_BYTES + cls.MAX_METADATA_BYTES:
                        raise ImageCheckpointError("checkpoint uncompressed data is too large")
        except ImageCheckpointError:
            raise
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
            raise ImageCheckpointError("image checkpoint is not a valid NPZ archive") from error

    @classmethod
    def load(cls, path: Path) -> TinyCanvasModel:
        """Load a strict v2 checkpoint or safely migrate the exact v1 layout."""

        path = Path(path)
        try:
            size = path.stat().st_size
        except OSError as error:
            raise ImageCheckpointError(f"cannot stat image checkpoint: {path}") from error
        if size <= 0 or size > cls.MAX_CHECKPOINT_BYTES:
            raise ImageCheckpointError("image checkpoint has an invalid compressed size")
        cls._preflight_checkpoint(path)
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "metadata" not in archive.files:
                    raise ImageCheckpointError("checkpoint metadata is missing")
                metadata = cls._parse_metadata(archive["metadata"])
                format_version = metadata.get("format_version")
                if type(format_version) is not int or format_version not in cls.SUPPORTED_FORMATS:
                    raise ImageCheckpointError("unsupported image checkpoint format")
                shape = cls._shape_from_metadata(metadata)
                expected_shapes = cls._expected_shapes(shape)
                if format_version == 1:
                    if set(metadata) != {"format_version", "shape"}:
                        raise ImageCheckpointError("v1 checkpoint metadata fields are invalid")
                    parameter_names = cls.V1_PARAMETER_NAMES
                else:
                    expected_metadata = {
                        "format_version",
                        "architecture",
                        "shape",
                        "styles",
                        "training",
                        "tensors",
                    }
                    if set(metadata) != expected_metadata:
                        raise ImageCheckpointError("v2 checkpoint metadata fields are invalid")
                    if metadata["architecture"] != "tinycanvas-coordinate-v2":
                        raise ImageCheckpointError("image checkpoint architecture is invalid")
                    if metadata["styles"] != list(cls.STYLES):
                        raise ImageCheckpointError("image checkpoint style order is invalid")
                    cls._validate_training_metadata(metadata["training"])
                    parameter_names = cls.PARAMETER_NAMES
                expected_files = {"metadata", *parameter_names}
                if set(archive.files) != expected_files:
                    raise ImageCheckpointError("checkpoint tensor set is invalid")

                tensors: dict[str, np.ndarray] = {}
                unpacked_bytes = 0
                tensor_metadata = metadata.get("tensors")
                if format_version == 2 and (
                    not isinstance(tensor_metadata, dict)
                    or set(tensor_metadata) != set(cls.PARAMETER_NAMES)
                ):
                    raise ImageCheckpointError("checkpoint tensor manifest is invalid")
                for name in parameter_names:
                    value = archive[name]
                    if value.dtype != np.dtype("float32"):
                        raise ImageCheckpointError(f"tensor {name} must use float32")
                    if value.shape != expected_shapes[name]:
                        raise ImageCheckpointError(f"tensor {name} has an invalid shape")
                    unpacked_bytes += value.nbytes
                    if unpacked_bytes > cls.MAX_UNPACKED_BYTES:
                        raise ImageCheckpointError("checkpoint tensors are too large")
                    copied = np.array(value, dtype=np.float32, copy=True, order="C")
                    if not np.isfinite(copied).all():
                        raise ImageCheckpointError(f"tensor {name} contains non-finite values")
                    if format_version == 2:
                        descriptor = tensor_metadata[name]
                        if not isinstance(descriptor, dict) or set(descriptor) != {
                            "shape",
                            "dtype",
                            "nbytes",
                            "sha256",
                        }:
                            raise ImageCheckpointError(f"tensor manifest entry {name} is invalid")
                        manifest_shape = descriptor["shape"]
                        if (
                            not isinstance(manifest_shape, list)
                            or any(type(dimension) is not int for dimension in manifest_shape)
                            or manifest_shape != list(expected_shapes[name])
                        ):
                            raise ImageCheckpointError(f"tensor manifest shape for {name} is invalid")
                        if descriptor["dtype"] != "float32":
                            raise ImageCheckpointError(f"tensor manifest dtype for {name} is invalid")
                        if (
                            type(descriptor["nbytes"]) is not int
                            or descriptor["nbytes"] != copied.nbytes
                        ):
                            raise ImageCheckpointError(f"tensor manifest byte count for {name} is invalid")
                        if descriptor["sha256"] != cls._tensor_digest(copied):
                            raise ImageCheckpointError(f"tensor hash mismatch for {name}")
                    tensors[name] = copied
        except ImageCheckpointError:
            raise
        except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
            raise ImageCheckpointError("image checkpoint could not be decoded safely") from error

        model = cls(shape)
        for name, value in tensors.items():
            setattr(model, name, value)
        if format_version == 1:
            model.style_embedding.fill(0.0)
            model.migrated_from = 1
        else:
            training = cls._validate_training_metadata(metadata["training"])
            model.training_steps = int(training["steps"])
            model.training_examples = int(training["examples"])
            initial_loss = training["initial_loss"]
            final_loss = training["final_loss"]
            model.training_initial_loss = None if initial_loss is None else float(initial_loss)
            model.training_final_loss = None if final_loss is None else float(final_loss)
            digest = training["manifest_sha256"]
            model.training_manifest_sha256 = None if digest is None else str(digest)
        return model

    def inspect(self) -> dict[str, object]:
        return {
            "architecture": "prompt-conditioned-coordinate-network-v2",
            "format_version": self.FORMAT_VERSION,
            "parameters": self.parameter_count,
            "shape": asdict(self.shape),
            "styles": list(self.STYLES),
            "quality_steps": dict(self.QUALITY_STEPS),
            "training": self._training_metadata(),
            "migrated_from": self.migrated_from,
            "output": f"128-{self.MAX_OUTPUT_SIZE} px tiled RGB stylised image",
        }


__all__ = ["ImageCheckpointError", "ImageShape", "TinyCanvasModel"]
