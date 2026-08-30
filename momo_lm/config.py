from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import default_home, ensure_runtime_dirs


@dataclass(slots=True)
class MomoConfig:
    home: Path
    model_path: Path
    image_model_path: Path
    database_path: Path
    agent_database_path: Path
    agent_workspace_path: Path
    mods_path: Path
    host: str = "127.0.0.1"
    port: int = 7860
    language: str = "zh-TW"
    self_learning: bool = True
    learning_rate: float = 0.0005
    temperature: float = 0.78
    top_k: int = 32
    max_new_tokens: int = 180
    max_crawl_pages: int = 8
    request_timeout: float = 10.0
    agent_approval_ttl_seconds: int = 900
    access_token: str | None = None
    allowed_hosts: list[str] = field(
        default_factory=lambda: ["127.0.0.1", "localhost", "::1"]
    )

    @classmethod
    def defaults(cls, home: Path | None = None) -> MomoConfig:
        root = (home or default_home()).expanduser().resolve()
        ensure_runtime_dirs(root)
        token = os.environ.get("MOMO_ACCESS_TOKEN")
        if token is not None:
            validate_access_token(token)
        return cls(
            home=root,
            model_path=root / "weights" / "momo-text-base.npz",
            image_model_path=root / "weights" / "momo-image-base.npz",
            database_path=root / "data" / "momo.db",
            agent_database_path=root / "data" / "agents.db",
            agent_workspace_path=root / "agent-workspace",
            mods_path=root / "mods",
            access_token=token,
        )

    @classmethod
    def load(cls, path: Path | None = None, home: Path | None = None) -> MomoConfig:
        config = cls.defaults(home)
        target = path or config.home / "config.json"
        if not target.exists():
            return config
        values: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
        path_fields = {
            "home",
            "model_path",
            "image_model_path",
            "database_path",
            "agent_database_path",
            "agent_workspace_path",
            "mods_path",
        }
        for key, value in values.items():
            if hasattr(config, key):
                setattr(config, key, Path(value).expanduser() if key in path_fields else value)
        if "agent_database_path" not in values:
            config.agent_database_path = config.home / "data" / "agents.db"
        if "agent_workspace_path" not in values:
            config.agent_workspace_path = config.home / "agent-workspace"
        environment_token = os.environ.get("MOMO_ACCESS_TOKEN")
        if environment_token is not None:
            config.access_token = environment_token
        config.validate()
        ensure_runtime_dirs(config.home)
        return config

    def save(self, path: Path | None = None) -> Path:
        target = path or self.home / "config.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        self.validate()
        values = asdict(self)
        for key in (
            "home",
            "model_path",
            "image_model_path",
            "database_path",
            "agent_database_path",
            "agent_workspace_path",
            "mods_path",
        ):
            values[key] = str(values[key])
        target.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        target.chmod(0o600)
        return target

    def validate(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must be a non-empty string")
        normalized_bind = self.host.strip().lower().strip("[]")
        if any(ord(character) < 33 or ord(character) > 126 for character in normalized_bind):
            raise ValueError("host must use visible ASCII")
        if "/" in normalized_bind or "@" in normalized_bind:
            raise ValueError("host must be a hostname or address without a port")
        if ":" in normalized_bind and normalized_bind not in {"::", "::1"}:
            raise ValueError("Only the :: and ::1 IPv6 bind addresses are supported")
        self.host = normalized_bind
        try:
            parsed_port = int(self.port)
        except (TypeError, ValueError) as exc:
            raise ValueError("port must be between 0 and 65535") from exc
        if isinstance(self.port, bool) or not 0 <= parsed_port <= 65_535:
            raise ValueError("port must be between 0 and 65535")
        self.port = parsed_port
        self.learning_rate = validate_training_rate(self.learning_rate, clamp=True)
        try:
            parsed_ttl = int(self.agent_approval_ttl_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "agent_approval_ttl_seconds must be between 30 and 86400"
            ) from exc
        if isinstance(self.agent_approval_ttl_seconds, bool) or not 30 <= parsed_ttl <= 86_400:
            raise ValueError("agent_approval_ttl_seconds must be between 30 and 86400")
        self.agent_approval_ttl_seconds = parsed_ttl
        if self.access_token is not None:
            validate_access_token(self.access_token)
        if not isinstance(self.allowed_hosts, list) or not self.allowed_hosts:
            raise ValueError("allowed_hosts must be a non-empty list")
        normalized_hosts: list[str] = []
        for host in self.allowed_hosts:
            if not isinstance(host, str) or not host.strip():
                raise ValueError("allowed_hosts entries must be non-empty strings")
            candidate = host.strip().lower().strip("[]")
            if any(ord(character) < 33 or ord(character) > 126 for character in candidate):
                raise ValueError("allowed_hosts entries must use visible ASCII")
            if (
                candidate == "*"
                or "/" in candidate
                or "@" in candidate
                or ":" in candidate
                and candidate != "::1"
            ):
                raise ValueError("allowed_hosts entries must be hostnames without ports")
            if candidate not in normalized_hosts:
                normalized_hosts.append(candidate)
        self.allowed_hosts = normalized_hosts


def validate_access_token(token: str) -> str:
    if not isinstance(token, str):
        raise ValueError("access_token must be a string")
    if not 1 <= len(token) <= 1024:
        raise ValueError("access_token must contain 1 to 1024 characters")
    if any(ord(character) < 33 or ord(character) > 126 for character in token):
        raise ValueError("access_token must use visible ASCII characters only")
    return token


def validate_training_rate(value: float, *, clamp: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError("learning_rate must be a number")
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("learning_rate must be a number") from exc
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if clamp:
        return max(0.0000001, min(rate, 0.005))
    if not 0.0000001 <= rate <= 0.005:
        raise ValueError("learning_rate must be between 1e-7 and 0.005")
    return rate
