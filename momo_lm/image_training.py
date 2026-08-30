from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import warnings
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .image_model import TinyCanvasModel


class ImageManifestError(ValueError):
    """Raised when an image-training manifest or one of its assets is invalid."""


@dataclass(frozen=True, slots=True)
class ImageTrainingExample:
    image_path: Path
    prompt: str
    style: str
    negative_prompt: str
    source: str
    license: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ImageTrainingManifest:
    path: Path
    sha256: str
    examples: tuple[ImageTrainingExample, ...]


@dataclass(frozen=True, slots=True)
class TrainingReport:
    initial_loss: float
    final_loss: float
    steps: int
    examples: int
    manifest_examples: int
    epochs: int
    samples_per_image: int
    manifest_sha256: str
    per_style_initial_loss: dict[str, float]
    per_style_final_loss: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable report."""

        return asdict(self)


MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_EXAMPLES = 10_000
MAX_TEXT_BYTES = 16 * 1024
SHA256_HEX_LENGTH = 64


def _read_bounded(path: Path, maximum: int, description: str) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ImageManifestError(f"cannot stat {description}: {path}") from error
    if size <= 0 or size > maximum:
        raise ImageManifestError(f"{description} has an invalid size")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ImageManifestError(f"cannot read {description}: {path}") from error
    if len(data) != size:
        raise ImageManifestError(f"{description} changed while it was being read")
    return data


def _required_text(record: dict[str, object], field: str, index: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ImageManifestError(f"examples[{index}].{field} must be a non-empty string")
    if len(value.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ImageManifestError(f"examples[{index}].{field} is too large")
    return value.strip()


def _optional_text(record: dict[str, object], field: str, index: int) -> str:
    value = record.get(field, "")
    if not isinstance(value, str):
        raise ImageManifestError(f"examples[{index}].{field} must be a string")
    if len(value.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ImageManifestError(f"examples[{index}].{field} is too large")
    return value.strip()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def load_manifest(path: Path) -> ImageTrainingManifest:
    """Load and authenticate a v1 local image-training manifest.

    Every example must declare its source, licence and exact SHA-256. Relative
    image paths are confined to the manifest directory after symlink resolution.
    """

    try:
        path = Path(path).resolve(strict=True)
    except OSError as error:
        raise ImageManifestError(f"training manifest is missing: {path}") from error
    raw = _read_bounded(path, MAX_MANIFEST_BYTES, "training manifest")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ImageManifestError("training manifest is not valid UTF-8 JSON") from error
    if not isinstance(document, dict) or set(document) != {"format_version", "examples"}:
        raise ImageManifestError("training manifest fields are invalid")
    if document["format_version"] != 1 or type(document["format_version"]) is not int:
        raise ImageManifestError("unsupported training manifest format")
    records = document["examples"]
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_EXAMPLES:
        raise ImageManifestError(f"manifest must contain 1 to {MAX_EXAMPLES} examples")

    root = path.parent
    examples: list[ImageTrainingExample] = []
    required = {"image", "prompt", "style", "source", "license", "sha256"}
    allowed = required | {"negative_prompt"}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not required <= set(record) or not set(record) <= allowed:
            raise ImageManifestError(f"examples[{index}] fields are invalid")
        image_name = _required_text(record, "image", index)
        candidate = Path(image_name)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ImageManifestError(
                f"examples[{index}].image must be a traversal-free relative path"
            )
        try:
            image_path = (root / candidate).resolve(strict=True)
            image_path.relative_to(root)
        except (OSError, ValueError) as error:
            raise ImageManifestError(
                f"examples[{index}].image escapes the manifest directory or is missing"
            ) from error
        image_bytes = _read_bounded(image_path, MAX_IMAGE_BYTES, f"examples[{index}].image")
        expected_hash = record["sha256"]
        if not _is_sha256(expected_hash):
            raise ImageManifestError(f"examples[{index}].sha256 must be lowercase SHA-256")
        actual_hash = hashlib.sha256(image_bytes).hexdigest()
        if actual_hash != expected_hash:
            raise ImageManifestError(f"examples[{index}].image SHA-256 mismatch")
        style = _required_text(record, "style", index)
        if style not in TinyCanvasModel.STYLES:
            raise ImageManifestError(f"examples[{index}].style is unsupported")
        _validate_image(image_bytes, index)
        examples.append(
            ImageTrainingExample(
                image_path=image_path,
                prompt=_required_text(record, "prompt", index),
                style=style,
                negative_prompt=_optional_text(record, "negative_prompt", index),
                source=_required_text(record, "source", index),
                license=_required_text(record, "license", index),
                sha256=actual_hash,
            )
        )
    return ImageTrainingManifest(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        examples=tuple(examples),
    )


def _validate_image(data: bytes, index: int) -> None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                    raise ImageManifestError(f"examples[{index}].image dimensions are invalid")
                if image.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ImageManifestError(
                        f"examples[{index}].image must be PNG, JPEG or WebP"
                    )
                image.verify()
    except ImageManifestError:
        raise
    except (
        OSError,
        SyntaxError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ImageManifestError(f"examples[{index}].image cannot be decoded safely") from error


def _load_rgb(example: ImageTrainingExample) -> np.ndarray:
    data = _read_bounded(example.image_path, MAX_IMAGE_BYTES, "training image")
    if hashlib.sha256(data).hexdigest() != example.sha256:
        raise ImageManifestError(f"training image SHA-256 changed: {example.image_path}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ImageManifestError(f"training image became unreadable: {example.image_path}") from error


def _samples(
    pixels: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width, _ = pixels.shape
    y_indices, x_indices = np.divmod(indices, width)
    x = (2.0 * (x_indices.astype(np.float32) + 0.5) / width - 1.0).astype(np.float32)
    y = (2.0 * (y_indices.astype(np.float32) + 0.5) / height - 1.0).astype(np.float32)
    coordinates = TinyCanvasModel.coordinate_features(x, y)
    targets = pixels[y_indices, x_indices]
    return coordinates, np.asarray(targets, dtype=np.float32)


def _evaluation_batches(
    manifest: ImageTrainingManifest, sample_count: int
) -> list[tuple[ImageTrainingExample, np.ndarray, np.ndarray]]:
    batches = []
    for example in manifest.examples:
        pixels = _load_rgb(example)
        count = min(sample_count, pixels.shape[0] * pixels.shape[1])
        indices = np.linspace(
            0,
            pixels.shape[0] * pixels.shape[1] - 1,
            num=count,
            dtype=np.int64,
        )
        coordinates, targets = _samples(pixels, indices)
        batches.append((example, coordinates, targets))
    return batches


def _evaluate(
    model: TinyCanvasModel,
    batches: list[tuple[ImageTrainingExample, np.ndarray, np.ndarray]],
) -> tuple[float, dict[str, float]]:
    style_losses: dict[str, list[float]] = {style: [] for style in TinyCanvasModel.STYLES}
    losses = []
    for example, coordinates, targets in batches:
        loss = model.loss(
            example.prompt,
            example.style,
            coordinates,
            targets,
            negative_prompt=example.negative_prompt,
        )
        losses.append(loss)
        style_losses[example.style].append(loss)
    per_style = {
        style: float(np.mean(values))
        for style, values in style_losses.items()
        if values
    }
    return float(np.mean(losses)), per_style


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _encode_png_rgb(pixels: np.ndarray) -> bytes:
    """Encode RGB pixels with stored DEFLATE blocks for byte-stable fixtures."""

    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("PNG pixels must have shape (height, width, 3)")
    height, width, _ = pixels.shape
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    blocks = bytearray(b"\x78\x01")
    for start in range(0, len(raw), 65_535):
        block = raw[start : start + 65_535]
        final = start + len(block) == len(raw)
        blocks.append(1 if final else 0)
        blocks.extend(struct.pack("<HH", len(block), 0xFFFF ^ len(block)))
        blocks.extend(block)
    blocks.extend(struct.pack(">I", zlib.adler32(raw) & 0xFFFFFFFF))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", bytes(blocks))
        + _png_chunk(b"IEND", b"")
    )


def create_reference_manifest(directory: Path, *, size: int = 64) -> Path:
    """Create the deterministic CC0 procedural set used for bundled weights."""

    if type(size) is not int or not 32 <= size <= 512:
        raise ValueError("size must be an integer in [32, 512]")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[0:size, 0:size]
    xf = (x + 0.5) / size
    yf = (y + 0.5) / size

    anime = np.empty((size, size, 3), dtype=np.float32)
    anime[..., 0] = 0.55 + 0.35 * (1.0 - yf)
    anime[..., 1] = 0.62 + 0.25 * xf
    anime[..., 2] = 0.88
    face = ((xf - 0.5) / 0.28) ** 2 + ((yf - 0.53) / 0.34) ** 2 < 1.0
    hair = face & ((yf < 0.38 + 0.09 * np.sin(xf * 18.0)) | (xf < 0.28))
    anime[face] = (1.0, 0.78, 0.68)
    anime[hair] = (0.17, 0.12, 0.30)
    eyes = (
        ((((xf - 0.40) / 0.038) ** 2 + ((yf - 0.53) / 0.025) ** 2) < 1.0)
        | ((((xf - 0.60) / 0.038) ** 2 + ((yf - 0.53) / 0.025) ** 2) < 1.0)
    )
    anime[eyes] = (0.08, 0.18, 0.38)
    anime[(np.abs(yf - 0.68) < 0.012) & (np.abs(xf - 0.5) < 0.07)] = (0.65, 0.18, 0.25)

    manga = np.ones((size, size, 3), dtype=np.float32) * 0.93
    panel = (np.abs(xf - 0.5) > 0.45) | (np.abs(yf - 0.5) > 0.45)
    ink_face = np.abs(((xf - 0.5) / 0.29) ** 2 + ((yf - 0.53) / 0.35) ** 2 - 1.0) < 0.06
    ink_hair = (yf < 0.38 + 0.08 * np.sin(xf * 20.0)) & (np.abs(xf - 0.5) < 0.3)
    hatch = ((x + 2 * y) % 7 == 0) & (yf > 0.72)
    manga[panel | ink_face | ink_hair | hatch] = 0.08
    manga[eyes] = 0.02

    illustration = np.empty((size, size, 3), dtype=np.float32)
    illustration[..., 0] = 0.95 - 0.35 * yf
    illustration[..., 1] = 0.52 + 0.25 * (1.0 - yf)
    illustration[..., 2] = 0.46 + 0.38 * (1.0 - yf)
    sun = (xf - 0.72) ** 2 + (yf - 0.27) ** 2 < 0.07**2
    back_hill = yf > 0.62 + 0.08 * np.sin(xf * 8.0)
    front_hill = yf > 0.76 + 0.06 * np.sin(xf * 13.0 + 1.0)
    illustration[sun] = (1.0, 0.92, 0.52)
    illustration[back_hill] = (0.28, 0.43, 0.38)
    illustration[front_hill] = (0.13, 0.28, 0.24)

    realistic = np.empty((size, size, 3), dtype=np.float32)
    realistic[..., 0] = 0.34 + 0.35 * (1.0 - yf)
    realistic[..., 1] = 0.49 + 0.32 * (1.0 - yf)
    realistic[..., 2] = 0.62 + 0.30 * (1.0 - yf)
    dx = (xf - 0.52) / 0.27
    dy = (yf - 0.52) / 0.27
    sphere = dx * dx + dy * dy < 1.0
    normal_z = np.sqrt(np.clip(1.0 - dx * dx - dy * dy, 0.0, 1.0))
    light = np.clip(-0.35 * dx - 0.45 * dy + 0.82 * normal_z, 0.0, 1.0)
    sphere_rgb = np.stack(
        [0.18 + 0.63 * light, 0.12 + 0.43 * light, 0.08 + 0.26 * light], axis=-1
    )
    realistic[sphere] = sphere_rgb[sphere]
    ground = yf > 0.79
    realistic[ground] = (0.22, 0.28, 0.20)

    references = {
        "anime": (anime, "anime portrait with clean cel shading"),
        "manga": (manga, "black and white manga portrait panel"),
        "illustration": (illustration, "painted sunset landscape illustration"),
        "realistic": (realistic, "realistic lit sphere in an outdoor scene"),
    }
    records = []
    for style, (pixels, prompt) in references.items():
        image_path = directory / f"reference-{style}.png"
        image_path.write_bytes(
            _encode_png_rgb((np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8))
        )
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        records.append(
            {
                "image": image_path.name,
                "prompt": prompt,
                "style": style,
                "negative_prompt": "watermark low quality",
                "source": f"momo-lm://procedural-reference/{style}",
                "license": "CC0-1.0",
                "sha256": digest,
            }
        )
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"format_version": 1, "examples": records},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def train_manifest(
    model: TinyCanvasModel,
    manifest: Path | ImageTrainingManifest,
    *,
    epochs: int = 1,
    learning_rate: float = 0.02,
    samples_per_image: int = 256,
    seed: int = 20260830,
    gradient_clip: float = 1.0,
) -> TrainingReport:
    """Train TinyCanvas from a validated manifest with deterministic NumPy SGD."""

    loaded = load_manifest(manifest) if isinstance(manifest, Path) else manifest
    if not isinstance(loaded, ImageTrainingManifest):
        raise TypeError("manifest must be a Path or ImageTrainingManifest")
    if type(epochs) is not int or not 1 <= epochs <= 10_000:
        raise ValueError("epochs must be an integer in [1, 10000]")
    if type(samples_per_image) is not int or not 1 <= samples_per_image <= 8192:
        raise ValueError("samples_per_image must be an integer in [1, 8192]")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if not math.isfinite(learning_rate) or not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be finite and in (0, 1]")
    if not math.isfinite(gradient_clip) or gradient_clip <= 0.0:
        raise ValueError("gradient_clip must be finite and positive")

    evaluation = _evaluation_batches(loaded, min(samples_per_image, 512))
    initial_loss, initial_styles = _evaluate(model, evaluation)
    images = [(example, _load_rgb(example)) for example in loaded.examples]
    rng = np.random.default_rng(seed)
    starting_steps = model.training_steps
    starting_examples = model.training_examples
    for _ in range(epochs):
        for selected in rng.permutation(len(images)):
            example, pixels = images[int(selected)]
            population = pixels.shape[0] * pixels.shape[1]
            count = min(samples_per_image, population)
            indices = rng.choice(population, size=count, replace=False)
            coordinates, targets = _samples(pixels, indices)
            model.train_batch(
                example.prompt,
                example.style,
                coordinates,
                targets,
                negative_prompt=example.negative_prompt,
                learning_rate=learning_rate,
                gradient_clip=gradient_clip,
            )
    final_loss, final_styles = _evaluate(model, evaluation)
    model.training_initial_loss = initial_loss
    model.training_final_loss = final_loss
    model.training_manifest_sha256 = loaded.sha256
    return TrainingReport(
        initial_loss=initial_loss,
        final_loss=final_loss,
        steps=model.training_steps - starting_steps,
        examples=model.training_examples - starting_examples,
        manifest_examples=len(loaded.examples),
        epochs=epochs,
        samples_per_image=samples_per_image,
        manifest_sha256=loaded.sha256,
        per_style_initial_loss=initial_styles,
        per_style_final_loss=final_styles,
    )


__all__ = [
    "ImageManifestError",
    "ImageTrainingExample",
    "ImageTrainingManifest",
    "TrainingReport",
    "create_reference_manifest",
    "load_manifest",
    "train_manifest",
]
