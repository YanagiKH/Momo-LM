from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import MomoConfig
from .runtime import MomoRuntime


class MomoLM:
    """Stable application API for embedding Momo-LM in local Python projects."""

    def __init__(self, config: MomoConfig | None = None, *, home: str | Path | None = None) -> None:
        if config is not None and home is not None:
            raise ValueError("provide either config or home, not both")
        self.runtime = MomoRuntime(config or MomoConfig.load(home=Path(home) if home is not None else None))
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        home: str | Path | None = None,
        *,
        config: MomoConfig | None = None,
    ) -> MomoLM:
        return cls(config, home=home)

    def chat(self, message: str, *, learn: bool | None = None) -> str:
        return str(self.runtime.chat(message, learn=learn)["response"])

    def chat_result(self, message: str, *, learn: bool | None = None) -> dict[str, Any]:
        return self.runtime.chat(message, learn=learn)

    def train(
        self,
        text: str,
        *,
        epochs: int = 3,
        learning_rate: float | None = None,
        source: str = "python-api",
    ) -> dict[str, Any]:
        return self.runtime.train(text, epochs=epochs, learning_rate=learning_rate, source=source)

    def ingest(self, text: str, *, source: str = "python-api", train: bool = False) -> dict[str, Any]:
        return self.runtime.ingest(text, source=source, train=train)

    def generate_image(
        self,
        prompt: str,
        output: str | Path,
        *,
        width: int = 512,
        height: int = 512,
        seed: int | None = None,
    ) -> Path:
        return self.runtime.generate_image(prompt, Path(output), width=width, height=height, seed=seed)

    def speak(self, text: str, output: str | Path, *, rate: int = 170) -> dict[str, Any]:
        return self.runtime.speech.synthesize(text, Path(output), rate=rate)

    def inspect(self) -> dict[str, Any]:
        return self.runtime.status()

    def close(self) -> None:
        if not self._closed:
            self.runtime.close()
            self._closed = True

    def __enter__(self) -> MomoLM:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __call__(self, message: str, *, learn: bool | None = None) -> str:
        return self.chat(message, learn=learn)


def load_model(
    home: str | Path | None = None,
    *,
    config: MomoConfig | None = None,
) -> MomoLM:
    return MomoLM.from_pretrained(home, config=config)
