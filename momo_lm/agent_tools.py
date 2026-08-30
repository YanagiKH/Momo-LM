from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runtime import MomoRuntime


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    capability: str
    mutating: bool
    description: str


TOOL_SPECS: dict[str, ToolSpec] = {
    "inspect_runtime": ToolSpec(
        "inspect_runtime", "runtime.inspect", False, "Inspect local model and runtime metadata"
    ),
    "search_knowledge": ToolSpec(
        "search_knowledge", "knowledge.read", False, "Search the local knowledge database"
    ),
    "list_files": ToolSpec(
        "list_files", "workspace.read", False, "List files inside the isolated agent workspace"
    ),
    "read_text_file": ToolSpec(
        "read_text_file", "workspace.read", False, "Read one UTF-8 file in the workspace"
    ),
    "draft_text": ToolSpec(
        "draft_text", "knowledge.read", False, "Create a deterministic local text draft"
    ),
    "training_guidance": ToolSpec(
        "training_guidance", "runtime.inspect", False, "Prepare a safe local training checklist"
    ),
    "write_text_file": ToolSpec(
        "write_text_file", "workspace.write", True, "Atomically write one workspace text file"
    ),
    "train_text": ToolSpec(
        "train_text", "model.train", True, "Train the local text checkpoint from supplied text"
    ),
}

# These classes of tools are deliberately not implemented. Keep the names explicit so
# callers receive a clear denial instead of accidentally discovering a future fallback.
FORBIDDEN_TOOL_NAMES = {
    "browse_web",
    "download",
    "drive_vehicle",
    "execute_program",
    "open_url",
    "run_command",
    "send_email",
    "shell",
    "use_camera",
    "use_microphone",
}


class AgentToolbox:
    """Small, capability-gated local tool registry with no network or device access."""

    def __init__(self, runtime: MomoRuntime, workspace: Path) -> None:
        self.runtime = runtime
        self.workspace = workspace.expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.workspace.chmod(0o700)
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "inspect_runtime": self._inspect_runtime,
            "search_knowledge": self._search_knowledge,
            "list_files": self._list_files,
            "read_text_file": self._read_text_file,
            "draft_text": self._draft_text,
            "training_guidance": self._training_guidance,
            "write_text_file": self._write_text_file,
            "train_text": self._train_text,
        }

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "capability": spec.capability,
                "mutating": spec.mutating,
                "description": spec.description,
            }
            for spec in TOOL_SPECS.values()
        ]

    def execute(
        self, name: str, arguments: dict[str, Any], *, capabilities: set[str]
    ) -> Any:
        if name in FORBIDDEN_TOOL_NAMES:
            raise PermissionError(f"Tool is forbidden: {name}")
        spec = TOOL_SPECS.get(name)
        handler = self._handlers.get(name)
        if spec is None or handler is None:
            raise ValueError(f"Unknown local agent tool: {name}")
        if spec.capability not in capabilities:
            raise PermissionError(f"Missing capability: {spec.capability}")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        return handler(arguments)

    def _safe_path(self, value: object, *, must_exist: bool = False) -> Path:
        raw = str(value).strip()
        if not raw or "\x00" in raw:
            raise ValueError("A non-empty workspace-relative path is required")
        candidate = Path(raw)
        if candidate.is_absolute():
            raise PermissionError("Absolute paths are not allowed")
        try:
            resolved = (self.workspace / candidate).resolve(strict=must_exist)
            resolved.relative_to(self.workspace)
        except (OSError, ValueError) as exc:
            raise PermissionError("Path escapes the isolated agent workspace") from exc
        return resolved

    def _inspect_runtime(self, _: dict[str, Any]) -> dict[str, Any]:
        status = self.runtime.status()
        return {
            "version": status.get("version"),
            "compute_backend": status.get("compute_backend"),
            "weights": status.get("weights"),
            "image_weights": status.get("image_weights"),
            "knowledge": status.get("knowledge"),
            "self_learning": status.get("self_learning"),
        }

    def _search_knowledge(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ValueError("Knowledge query is empty")
        limit = max(1, min(int(arguments.get("limit", 4)), 20))
        hits = self.runtime.store.search(query[:12_000], limit=limit)
        return {
            "query": query,
            "hits": [
                {"source": hit.source, "content": hit.content[:4_000], "score": hit.score}
                for hit in hits
            ],
        }

    def _list_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = str(arguments.get("path", "."))
        root = self._safe_path(relative, must_exist=True)
        if not root.is_dir():
            raise ValueError("Workspace path is not a directory")
        limit = max(1, min(int(arguments.get("limit", 100)), 500))
        files: list[str] = []
        for candidate in sorted(root.rglob("*")):
            if len(files) >= limit:
                break
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(self.workspace)
            except (OSError, ValueError):
                continue
            if resolved.is_file():
                files.append(resolved.relative_to(self.workspace).as_posix())
        return {"path": root.relative_to(self.workspace).as_posix() or ".", "files": files}

    def _read_text_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._safe_path(arguments.get("path", ""), must_exist=True)
        if not path.is_file():
            raise ValueError("Workspace path is not a file")
        size = path.stat().st_size
        if size > 262_144:
            raise ValueError("Text file exceeds 256 KiB")
        raw = path.read_bytes()
        if b"\x00" in raw:
            raise ValueError("Binary files cannot be read by an agent")
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Agent files must be UTF-8 text") from exc
        return {"path": path.relative_to(self.workspace).as_posix(), "content": content}

    @staticmethod
    def _draft_text(arguments: dict[str, Any]) -> dict[str, Any]:
        goal = " ".join(str(arguments.get("goal", "")).split())
        profile = str(arguments.get("profile", "workplace"))
        if not goal:
            raise ValueError("Draft goal is empty")
        lead = {
            "coding": "Implementation note",
            "copilot": "Copilot brief",
            "training": "Training note",
            "workplace": "Workplace draft",
        }.get(profile, "Draft")
        return {
            "draft": f"{lead}: {goal}",
            "note": "This deterministic draft is local and has not been sent or published.",
        }

    @staticmethod
    def _training_guidance(arguments: dict[str, Any]) -> dict[str, Any]:
        goal = " ".join(str(arguments.get("goal", "")).split())
        return {
            "goal": goal,
            "checklist": [
                "Use data you are authorized to process.",
                "Keep a validation set separate from training data.",
                "Start with a reversible checkpoint and a small learning rate.",
                "Compare measured validation loss before accepting new weights.",
            ],
        }

    def _write_text_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._safe_path(arguments.get("path", ""), must_exist=False)
        content = str(arguments.get("content", ""))
        encoded = content.encode("utf-8")
        if len(encoded) > 1_048_576:
            raise ValueError("Agent write exceeds 1 MiB")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Resolve the newly-created parent again after mkdir to reject symlink races.
        try:
            path.parent.resolve(strict=True).relative_to(self.workspace)
        except (OSError, ValueError) as exc:
            raise PermissionError("Write parent escapes the isolated workspace") from exc
        current = path.parent
        while current != self.workspace:
            current.chmod(0o700)
            current = current.parent
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "path": path.relative_to(self.workspace).as_posix(),
            "bytes_written": len(encoded),
        }

    def _train_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        text = str(arguments.get("text", ""))
        if not text.strip():
            raise ValueError("Training text is empty")
        if len(text) > 100_000:
            raise ValueError("Agent training input exceeds 100,000 characters")
        epochs = max(1, min(int(arguments.get("epochs", 1)), 10))
        source = str(arguments.get("source", "agent"))[:500]
        return self.runtime.train(text, epochs=epochs, source=source)
