from __future__ import annotations

import os
import sys
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
