from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path


def package_root() -> Path:
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return Path(bundled) / "momo_lm"
    return Path(__file__).resolve().parent


def default_home() -> Path:
    override = os.environ.get("MOMO_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".momo-lm"


def ensure_runtime_dirs(home: Path) -> None:
    for name in ("weights", "data", "generated", "speech", "mods"):
        (home / name).mkdir(parents=True, exist_ok=True)


def atomic_copy(source: Path, target: Path) -> Path:
    """Copy a file through a same-directory temporary and atomically replace it."""

    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        if os.name != "nt":
            directory_descriptor = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        return target
    finally:
        temporary.unlink(missing_ok=True)
