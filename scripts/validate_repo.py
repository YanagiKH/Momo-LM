from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from momo_lm.image_model import TinyCanvasModel  # noqa: E402
from momo_lm.model import NeuralTextModel  # noqa: E402


def fail(message: str) -> None:
    raise SystemExit(f"validation failed: {message}")


def main() -> None:
    required = [
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "momo_lm/web/index.html",
        "momo_lm/web/app.css",
        "momo_lm/web/app.js",
        "momo_lm/assets/weights/momo-text-base.npz",
        "momo_lm/assets/weights/momo-image-base.npz",
        "momo_lm/api.py",
        "momo_lm/backend.py",
        "MomoLM.py",
        "setup.py",
        "native/CMakeLists.txt",
        "native/include/momo_core.h",
        "native/src/tensor.c",
        "native/src/runtime.cpp",
        "native/python/module.cpp",
        "native/rust/Cargo.toml",
        "native/rust/Cargo.lock",
        "native/rust/src/lib.rs",
        "docs/NATIVE_CORE.md",
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
        "installer/windows.iss",
        "installer/languages/ChineseTraditional.isl",
        "installer/languages/Japanese.isl",
        "installer/languages/LICENSE-Inno-Setup.txt",
        "installer/languages/README.md",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            fail(f"missing {relative}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    local_links = re.findall(r"(?:src|href)=?[\"']?([^\"'> )]+)|\]\(([^)]+)\)", readme)
    for pair in local_links:
        target = next((part for part in pair if part), "").split("#", 1)[0]
        if target and not re.match(r"(?:https?:|#|mailto:)", target) and not (ROOT / target).exists():
            fail(f"README references missing path: {target}")

    for svg in (ROOT / "docs" / "assets").glob("*.svg"):
        ET.parse(svg)
    json.loads((ROOT / "config.example.json").read_text(encoding="utf-8"))

    installer = (ROOT / "installer/windows.iss").read_text(encoding="utf-8")
    language_hashes = {
        "ChineseTraditional.isl": "0d68bc0f4fd0acdba43b43531f5a7bf1fd0efbcbd5c3e1b9b41fc096991950db",
        "Japanese.isl": "8b610f3f6b55707c9629e1009e700ec963015d900d03db65c8d69680e3b55394",
    }
    for filename, expected_hash in language_hashes.items():
        messages_file = f'MessagesFile: "{{#SourcePath}}\\languages\\{filename}"'
        if messages_file not in installer:
            fail(f"Windows installer does not use vendored {filename}")
        language_file = ROOT / "installer/languages" / filename
        actual_hash = hashlib.sha256(language_file.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            fail(f"unexpected checksum for installer/languages/{filename}: {actual_hash}")

    text_model = NeuralTextModel.load(ROOT / "momo_lm/assets/weights/momo-text-base.npz")
    image_model = TinyCanvasModel.load(ROOT / "momo_lm/assets/weights/momo-image-base.npz")
    if text_model.parameter_count != 184_131:
        fail(f"unexpected text model parameter count: {text_model.parameter_count}")
    if text_model.inspect()["format_version"] != 2:
        fail("text checkpoint migration did not produce model format 2")
    with np.load(ROOT / "momo_lm/assets/weights/momo-text-base.npz", allow_pickle=False) as archive:
        if json.loads(str(archive["metadata"])).get("format_version") != 2:
            fail("bundled text checkpoint is not stored in model format 2")
    if image_model.inspect()["parameters"] < 1_000:
        fail("image checkpoint is incomplete")
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "smoke.png"
        image_model.generate("validation", 128, 128, seed=1).save(output)
        if output.stat().st_size < 1_000:
            fail("generated image is unexpectedly small")
    print("repository validation passed")


if __name__ == "__main__":
    main()
