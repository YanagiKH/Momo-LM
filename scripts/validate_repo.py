from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The recorded checkpoint metrics use the deterministic NumPy reference path.
# Native availability is verified separately before this script runs in CI.
os.environ["MOMO_BACKEND"] = "numpy"
os.environ.pop("MOMO_REQUIRE_NATIVE", None)

import momo_lm  # noqa: E402
from momo_lm.evaluation import evaluate_text  # noqa: E402
from momo_lm.image_model import TinyCanvasModel  # noqa: E402
from momo_lm.image_training import create_reference_manifest, load_manifest  # noqa: E402
from momo_lm.model import NeuralTextModel  # noqa: E402


def fail(message: str) -> None:
    raise SystemExit(f"validation failed: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        fail(f"unexpected {label}: {actual!r} != {expected!r}")


def require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=2e-6, abs_tol=2e-7):
        fail(f"unexpected {label}: {actual!r} != {expected!r}")


def validate_required_files() -> None:
    required = [
        "README.md",
        "LICENSE",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "VERSION",
        "pyproject.toml",
        "MomoLM.py",
        "setup.py",
        "evals/base-model-report.json",
        "evals/README.md",
        "momo_lm/agent.py",
        "momo_lm/agent_store.py",
        "momo_lm/agent_tools.py",
        "momo_lm/api.py",
        "momo_lm/backend.py",
        "momo_lm/evaluation.py",
        "momo_lm/image_training.py",
        "momo_lm/training.py",
        "momo_lm/web/index.html",
        "momo_lm/web/app.css",
        "momo_lm/web/app.js",
        "momo_lm/assets/weights/momo-text-base.npz",
        "momo_lm/assets/weights/momo-image-base.npz",
        "native/CMakeLists.txt",
        "native/include/momo_core.h",
        "native/src/attention.c",
        "native/src/tensor.c",
        "native/src/runtime.cpp",
        "native/python/module.cpp",
        "native/rust/Cargo.toml",
        "native/rust/Cargo.lock",
        "native/rust/src/lib.rs",
        "docs/AGENTS.md",
        "docs/BRANCH_PROTECTION.md",
        "docs/IMAGE_TRAINING.md",
        "docs/NATIVE_CORE.md",
        "docs/TRAINING.md",
        ".github/CODEOWNERS",
        ".github/dependabot.yml",
        ".github/workflows/ci.yml",
        ".github/workflows/codeql.yml",
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


def validate_docs_and_metadata() -> dict[str, Any]:
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    require_equal(project["project"]["version"], version, "pyproject version")
    require_equal(momo_lm.__version__, version, "Python package version")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    local_links = re.findall(r"(?:src|href)=?[\"']?([^\"'> )]+)|\]\(([^)]+)\)", readme)
    for pair in local_links:
        target = next((part for part in pair if part), "").split("#", 1)[0]
        if target and not re.match(r"(?:https?:|#|mailto:)", target) and not (ROOT / target).exists():
            fail(f"README references missing path: {target}")

    for svg in (ROOT / "docs" / "assets").glob("*.svg"):
        ET.parse(svg)
    json.loads((ROOT / "config.example.json").read_text(encoding="utf-8"))
    report = json.loads((ROOT / "evals/base-model-report.json").read_text(encoding="utf-8"))
    require_equal(report["project_version"], version, "evaluation report version")

    for workflow in (ROOT / ".github/workflows").glob("*.yml"):
        content = workflow.read_text(encoding="utf-8")
        uses = re.findall(r"^\s*-\s*uses:\s*([^@\s]+)@([^\s#]+)", content, flags=re.MULTILINE)
        for action, reference in uses:
            if not re.fullmatch(r"[0-9a-f]{40}", reference):
                fail(f"{workflow.relative_to(ROOT)} does not pin {action} to a 40-character SHA")

    installer = (ROOT / "installer/windows.iss").read_text(encoding="utf-8")
    language_hashes = {
        "ChineseTraditional.isl": "0d68bc0f4fd0acdba43b43531f5a7bf1fd0efbcbd5c3e1b9b41fc096991950db",
        "Japanese.isl": "8b610f3f6b55707c9629e1009e700ec963015d900d03db65c8d69680e3b55394",
    }
    for filename, expected_hash in language_hashes.items():
        messages_file = f'MessagesFile: "{{#SourcePath}}\\languages\\{filename}"'
        if messages_file not in installer:
            fail(f"Windows installer does not use vendored {filename}")
        actual_hash = sha256_file(ROOT / "installer/languages" / filename)
        require_equal(actual_hash, expected_hash, f"installer/languages/{filename} SHA-256")

    backups = list((ROOT / "momo_lm").rglob("*.npz.last-good"))
    if backups:
        fail(f"runtime checkpoint backups are present in the package tree: {backups}")
    return report


def validate_text_checkpoint(report: dict[str, Any]) -> None:
    text_report = report["text"]
    checkpoint = ROOT / text_report["checkpoint"]
    require_equal(sha256_file(checkpoint), text_report["checkpoint_sha256"], "text checkpoint SHA-256")
    model = NeuralTextModel.load(checkpoint, recover=False)
    inspection = model.inspect()
    require_equal(inspection["format_version"], text_report["format_version"], "text format")
    require_equal(inspection["parameters"], text_report["parameters"], "text parameter count")
    require_equal(inspection["shape"], text_report["shape"], "text shape")
    require_equal(len(inspection["layers"]), text_report["tensors"], "text tensor count")
    require_equal(
        inspection["training_steps"],
        text_report["training"]["optimizer_steps"],
        "text optimizer steps",
    )
    require_equal(
        inspection["tokens_seen"], text_report["training"]["targets_seen"], "text targets seen"
    )
    training = inspection["training"]
    require_equal(training["validation_is_held_out"], False, "text held-out flag")
    require_equal(
        training["provenance"]["validation_protocol"],
        "starter-corpus-overlap-not-held-out",
        "text validation protocol",
    )

    corpus_path = ROOT / text_report["dataset"]["path"]
    corpus_bytes = corpus_path.read_bytes()
    require_equal(
        hashlib.sha256(corpus_bytes).hexdigest(),
        text_report["dataset"]["sha256"],
        "text corpus SHA-256",
    )
    require_equal(
        len(corpus_bytes), text_report["dataset"]["utf8_bytes"], "text corpus byte count"
    )
    corpus = corpus_bytes.decode("utf-8")
    baseline = evaluate_text(NeuralTextModel(), corpus, batch_size=256)
    trained = evaluate_text(model, corpus, batch_size=256)
    for name in ("negative_log_likelihood", "perplexity", "top1_accuracy"):
        require_close(
            getattr(baseline, name),
            float(text_report["metrics"]["baseline"][name]),
            f"text baseline {name}",
        )
        require_close(
            getattr(trained, name),
            float(text_report["metrics"]["trained"][name]),
            f"text trained {name}",
        )
    if not trained.negative_log_likelihood < baseline.negative_log_likelihood:
        fail("bundled text training did not reduce fixed-corpus negative log-likelihood")


def validate_image_checkpoint(report: dict[str, Any]) -> None:
    image_report = report["image"]
    checkpoint = ROOT / image_report["checkpoint"]
    require_equal(
        sha256_file(checkpoint), image_report["checkpoint_sha256"], "image checkpoint SHA-256"
    )
    model = TinyCanvasModel.load(checkpoint)
    inspection = model.inspect()
    require_equal(inspection["format_version"], image_report["format_version"], "image format")
    require_equal(inspection["parameters"], image_report["parameters"], "image parameter count")
    require_equal(inspection["styles"], image_report["shape"]["styles"], "image styles")
    for key in ("prompt_features", "latent_size", "hidden_size"):
        require_equal(inspection["shape"][key], image_report["shape"][key], f"image shape {key}")
    training = inspection["training"]
    require_equal(
        training["steps"], image_report["training"]["optimizer_steps"], "image optimizer steps"
    )
    require_equal(
        training["examples"], image_report["training"]["examples_processed"], "image examples"
    )
    require_equal(
        training["manifest_sha256"],
        image_report["dataset"]["manifest_sha256"],
        "image manifest SHA-256",
    )
    require_close(
        float(training["initial_loss"]),
        float(image_report["metrics"]["baseline_mean_loss"]),
        "image initial loss",
    )
    require_close(
        float(training["final_loss"]),
        float(image_report["metrics"]["trained_mean_loss"]),
        "image final loss",
    )
    if not float(training["final_loss"]) < float(training["initial_loss"]):
        fail("bundled image training did not reduce fixed-coordinate loss")

    with tempfile.TemporaryDirectory(prefix="momo-validation-images-") as directory:
        manifest_path = create_reference_manifest(Path(directory))
        manifest = load_manifest(manifest_path)
        require_equal(
            manifest.sha256,
            image_report["dataset"]["manifest_sha256"],
            "generated image manifest SHA-256",
        )
        for index, style in enumerate(TinyCanvasModel.STYLES):
            output = Path(directory) / f"smoke-{style}.png"
            model.generate(
                f"validation {style}",
                128,
                128,
                seed=index + 1,
                style=style,
                negative_prompt="artifact",
                quality="draft",
            ).save(output, format="PNG")
            if output.stat().st_size < 1_000:
                fail(f"generated {style} image is unexpectedly small")


def main() -> None:
    validate_required_files()
    report = validate_docs_and_metadata()
    validate_text_checkpoint(report)
    validate_image_checkpoint(report)
    print("repository validation passed")


if __name__ == "__main__":
    main()
