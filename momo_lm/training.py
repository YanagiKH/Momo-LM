from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .evaluation import EvaluationMetrics, evaluate_text
from .tokenizer import ByteTokenizer

if TYPE_CHECKING:
    from .model import NeuralTextModel


@dataclass(frozen=True, slots=True)
class AdamWConfig:
    epochs: int = 1
    learning_rate: float = 0.002
    batch_size: int = 144
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 20260803
    max_replay_characters: int = 64_000

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "seed", "max_replay_characters"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        for name in (
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "weight_decay",
            "gradient_clip",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
        if not 1 <= self.epochs <= 10_000:
            raise ValueError("epochs must be between 1 and 10,000")
        if not 1 <= self.batch_size <= 65_536:
            raise ValueError("batch_size must be between 1 and 65,536")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be finite and in (0, 1]")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW beta values must be in [0, 1)")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be finite and positive")
        if not 0 <= self.max_replay_characters <= 10_000_000:
            raise ValueError("max_replay_characters is outside the supported range")
        if not 0 <= self.seed < 2**64:
            raise ValueError("seed must fit uint64")


@dataclass(slots=True)
class TrainingResult:
    epoch_losses: list[float]
    optimizer_steps: int
    targets_seen: int
    training_targets: int
    replay_documents: int
    validation: EvaluationMetrics | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch_losses": list(self.epoch_losses),
            "optimizer_steps": self.optimizer_steps,
            "targets_seen": self.targets_seen,
            "training_targets": self.training_targets,
            "replay_documents": self.replay_documents,
            "validation": self.validation.to_dict() if self.validation else None,
        }


class AdamWTrainer:
    """Deterministic mini-batch AdamW with bounded replay and gradient clipping."""

    def __init__(self, model: NeuralTextModel, config: AdamWConfig | None = None) -> None:
        self.model = model
        self.config = config or AdamWConfig(seed=model.shape.seed)
        self._first = {name: np.zeros_like(value) for name, value in model.parameters().items()}
        self._second = {name: np.zeros_like(value) for name, value in model.parameters().items()}
        self._optimizer_step = 0

    def _training_stream(self, text: str, replay_texts: Sequence[str]) -> tuple[str, int]:
        if not isinstance(text, str):
            raise TypeError("training text must be a string")
        if not text.strip():
            return "", 0
        primary = text
        remaining = self.config.max_replay_characters
        selected: list[str] = []
        for replay in replay_texts:
            if not isinstance(replay, str):
                raise TypeError("replay documents must be strings")
            cleaned = replay.strip()
            if not cleaned or remaining <= 0:
                continue
            clipped = cleaned[:remaining]
            selected.append(clipped)
            remaining -= len(clipped)
        stream = primary
        if selected:
            stream += "\n" + "\n".join(selected)
        return stream, len(selected)

    @staticmethod
    def _uses_weight_decay(name: str, value: np.ndarray) -> bool:
        return value.ndim >= 2 and name not in {"embedding", "position_embedding"}

    def _apply(self, gradients: dict[str, np.ndarray]) -> float:
        squared_norm = sum(float(np.sum(gradient.astype(np.float64) ** 2)) for gradient in gradients.values())
        norm = math.sqrt(squared_norm)
        scale = min(1.0, self.config.gradient_clip / max(norm, 1e-12))
        self._optimizer_step += 1
        correction1 = 1.0 - self.config.beta1**self._optimizer_step
        correction2 = 1.0 - self.config.beta2**self._optimizer_step
        for name, parameter in self.model.parameters().items():
            gradient = np.asarray(gradients[name] * scale, dtype=np.float32)
            if not np.all(np.isfinite(gradient)):
                raise FloatingPointError(f"non-finite gradient in {name}")
            first = self._first[name]
            second = self._second[name]
            first *= self.config.beta1
            first += (1.0 - self.config.beta1) * gradient
            second *= self.config.beta2
            second += (1.0 - self.config.beta2) * gradient * gradient
            update = (first / correction1) / (np.sqrt(second / correction2) + self.config.epsilon)
            if self._uses_weight_decay(name, parameter):
                parameter *= 1.0 - self.config.learning_rate * self.config.weight_decay
            parameter -= self.config.learning_rate * update
            if not np.all(np.isfinite(parameter)):
                raise FloatingPointError(f"optimizer produced non-finite parameter {name}")
        return norm

    def fit(
        self,
        text: str,
        *,
        replay_texts: Sequence[str] = (),
        validation_text: str | None = None,
        validation_is_held_out: bool = False,
    ) -> TrainingResult:
        if not isinstance(validation_is_held_out, bool):
            raise TypeError("validation_is_held_out must be a boolean")
        stream, replay_count = self._training_stream(text, replay_texts)
        if not stream:
            return TrainingResult([], 0, 0, 0, replay_count, None)
        tokens = ByteTokenizer().encode(stream, bos=True, eos=True)
        context_stream, targets = self.model._context_stream(tokens)
        rng = np.random.default_rng(self.config.seed)
        epoch_losses: list[float] = []
        initial_steps = self.model.steps
        for _ in range(self.config.epochs):
            order = rng.permutation(len(targets))
            cumulative_loss = 0.0
            for start in range(0, len(order), self.config.batch_size):
                selection = order[start : start + self.config.batch_size]
                contexts, selected_targets = self.model._context_batch(
                    context_stream, targets, selection
                )
                loss, gradients = self.model._loss_and_gradients(
                    contexts, selected_targets
                )
                self._apply(gradients)
                cumulative_loss += loss * len(selection)
                self.model.steps += 1
            self.model.tokens_seen += len(targets)
            epoch_losses.append(cumulative_loss / len(targets))

        validation = (
            evaluate_text(self.model, validation_text, batch_size=self.config.batch_size)
            if validation_text is not None
            else None
        )
        self.model.training_metadata = {
            "optimizer": "adamw",
            "optimizer_config": asdict(self.config),
            "last_run": {
                "epoch_losses": [float(value) for value in epoch_losses],
                "optimizer_steps": self.model.steps - initial_steps,
                "training_targets_per_epoch": int(len(targets)),
                "targets_seen": int(len(targets) * self.config.epochs),
                "replay_documents": replay_count,
            },
            "validation": validation.to_dict() if validation else None,
            "validation_is_held_out": validation_is_held_out if validation is not None else None,
            "deterministic": True,
        }
        return TrainingResult(
            epoch_losses=epoch_losses,
            optimizer_steps=self.model.steps - initial_steps,
            targets_seen=len(targets) * self.config.epochs,
            training_targets=len(targets),
            replay_documents=replay_count,
            validation=validation,
        )
