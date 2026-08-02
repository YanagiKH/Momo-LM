from __future__ import annotations

import importlib.util
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

LOGGER = logging.getLogger("momo_lm.mods")
COMMAND = re.compile(r"^/[a-zA-Z][a-zA-Z0-9_-]*")


@dataclass(slots=True)
class ModSpec:
    name: str
    version: str = "0.1.0"
    description: str = ""
    commands: dict[str, Callable[[str], str]] = field(default_factory=dict)
    before_chat: Callable[[str], str] | None = None
    after_chat: Callable[[str, str], str] | None = None


class ModManager:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mods: list[ModSpec] = []
        self.errors: list[dict[str, str]] = []

    def load(self) -> None:
        self.mods.clear()
        self.errors.clear()
        for path in sorted(self.directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                module = self._load_module(path)
                register = getattr(module, "register", None)
                if not callable(register):
                    raise ValueError("mod must export register()")
                spec = register()
                if not isinstance(spec, ModSpec):
                    raise TypeError("register() must return momo_lm.mods.ModSpec")
                self.mods.append(spec)
            except Exception as exc:  # mods are isolated so one failure cannot stop the app
                LOGGER.exception("Failed to load mod %s", path.name)
                self.errors.append({"file": path.name, "error": str(exc)})

    @staticmethod
    def _load_module(path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(f"momo_user_mod_{path.stem}", path)
        if not spec or not spec.loader:
            raise ImportError(f"Cannot create import specification for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def process_before(self, text: str) -> str:
        for mod in self.mods:
            if mod.before_chat:
                text = mod.before_chat(text)
        return text

    def process_after(self, text: str, response: str) -> str:
        for mod in self.mods:
            if mod.after_chat:
                response = mod.after_chat(text, response)
        return response

    def command(self, text: str) -> str | None:
        match = COMMAND.match(text)
        if not match:
            return None
        command = match.group(0).lower()
        argument = text[match.end() :].strip()
        for mod in self.mods:
            handler = mod.commands.get(command)
            if handler:
                return str(handler(argument))
        return None

    def status(self) -> dict[str, Any]:
        return {
            "loaded": [
                {"name": mod.name, "version": mod.version, "description": mod.description, "commands": sorted(mod.commands)}
                for mod in self.mods
            ],
            "errors": self.errors,
        }
