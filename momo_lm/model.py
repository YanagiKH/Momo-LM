from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .backend import TensorBackend, get_backend
from .tokenizer import ByteTokenizer


@dataclass(slots=True)
class ModelShape:
    vocab_size: int = ByteTokenizer.vocab_size
    context_length: int = 24
    embedding_size: int = 32
    hidden_size: int = 96
    neuron_group_size: int = 16
    seed: int = 20260803


class NeuralTextModel:
    """A compact byte-level neural language model with inspectable backpropagation.

    The network combines ordered token embeddings, gated mixed-activation neuron
    groups, a residual context path and a next-byte distribution. Matrix-heavy
    operations automatically use the Rust or C/C++ core when it is available.
    """

    FORMAT_VERSION = 2
    SUPPORTED_FORMATS = {1, 2}

    def __init__(self, shape: ModelShape | None = None) -> None:
        self.shape = shape or ModelShape()
        rng = np.random.default_rng(self.shape.seed)
        e, c, h, v = (
            self.shape.embedding_size,
            self.shape.context_length,
            self.shape.hidden_size,
            self.shape.vocab_size,
        )
        self.embedding = rng.normal(0, 0.08, (v, e)).astype(np.float32)
        self.w_hidden = rng.normal(0, math.sqrt(2 / (c * e)), (c * e, h)).astype(np.float32)
        self.b_hidden = np.zeros(h, dtype=np.float32)
        self.w_gate = np.zeros((c * e, h), dtype=np.float32)
        self.b_gate = np.full(h, 4.0, dtype=np.float32)
        self.w_residual = np.zeros((e, h), dtype=np.float32)
        self.w_output = rng.normal(0, math.sqrt(2 / h), (h, v)).astype(np.float32)
        self.b_output = np.zeros(v, dtype=np.float32)
        self.steps = 0
        self.tokens_seen = 0
        self.backend: TensorBackend = get_backend()

    @property
    def parameter_count(self) -> int:
        return sum(value.size for value in self.parameters().values())

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "embedding": self.embedding,
            "w_hidden": self.w_hidden,
            "b_hidden": self.b_hidden,
            "w_gate": self.w_gate,
            "b_gate": self.b_gate,
            "w_residual": self.w_residual,
            "w_output": self.w_output,
            "b_output": self.b_output,
        }

    def _contexts(self, tokens: list[int]) -> tuple[np.ndarray, np.ndarray]:
        c = self.shape.context_length
        stream = [ByteTokenizer.PAD] * c + tokens
        x = np.asarray([stream[index : index + c] for index in range(len(tokens))], dtype=np.int32)
        y = np.asarray(tokens, dtype=np.int32)
        return x, y

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        positive = value >= 0
        result = np.empty_like(value)
        result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
        exponential = np.exp(value[~positive])
        result[~positive] = exponential / (1.0 + exponential)
        return result

    def _activate(self, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        activated = np.empty_like(value)
        derivative = np.empty_like(value)
        size = self.shape.neuron_group_size
        coefficient = math.sqrt(2.0 / math.pi)
        for start in range(0, value.shape[1], size):
            stop = min(start + size, value.shape[1])
            group = start // size
            chunk = value[:, start:stop]
            if group % 3 == 0:
                output = np.tanh(chunk)
                activated[:, start:stop] = output
                derivative[:, start:stop] = 1.0 - output * output
            elif group % 3 == 1:
                cubic = chunk**3
                transformed = coefficient * (chunk + 0.044715 * cubic)
                tangent = np.tanh(transformed)
                activated[:, start:stop] = 0.5 * chunk * (1.0 + tangent)
                derivative[:, start:stop] = 0.5 * (1.0 + tangent) + 0.5 * chunk * (
                    1.0 - tangent * tangent
                ) * coefficient * (1.0 + 3.0 * 0.044715 * chunk * chunk)
            else:
                sigmoid = self._sigmoid(chunk)
                activated[:, start:stop] = chunk * sigmoid
                derivative[:, start:stop] = sigmoid + chunk * sigmoid * (1.0 - sigmoid)
        return activated, derivative

    def _forward(
        self, contexts: np.ndarray, *, inference: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        embedded = self.embedding[contexts]
        flat = embedded.reshape(contexts.shape[0], -1)
        residual_input = embedded.mean(axis=1)
        if inference:
            hidden = self.backend.neuron_group(
                flat,
                self.w_hidden,
                self.b_hidden,
                self.w_gate,
                self.b_gate,
                residual_input,
                self.w_residual,
                self.shape.neuron_group_size,
            )
            cache: dict[str, np.ndarray] = {}
        else:
            projection = self.backend.matmul(flat, self.w_hidden) + self.b_hidden
            gate_pre = self.backend.matmul(flat, self.w_gate) + self.b_gate
            gate = self._sigmoid(gate_pre)
            activated, activation_derivative = self._activate(projection)
            hidden = activated * gate + self.backend.matmul(residual_input, self.w_residual)
            cache = {
                "embedded": embedded,
                "residual_input": residual_input,
                "gate": gate,
                "activated": activated,
                "activation_derivative": activation_derivative,
            }
        scores = self.backend.matmul(hidden, self.w_output) + self.b_output
        return scores, hidden, flat, cache

    def logits(
        self, contexts: np.ndarray, *, inference: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, hidden, flat, _ = self._forward(contexts, inference=inference)
        return scores, hidden, flat

    def train_text(
        self,
        text: str,
        *,
        epochs: int = 1,
        learning_rate: float = 0.025,
        batch_size: int = 128,
        seed: int | None = None,
    ) -> list[float]:
        tokenizer = ByteTokenizer()
        tokens = tokenizer.encode(text, bos=True, eos=True)
        if len(tokens) < 2:
            return []
        contexts, targets = self._contexts(tokens)
        rng = np.random.default_rng(seed if seed is not None else self.steps + self.shape.seed)
        losses: list[float] = []
        for _ in range(max(1, epochs)):
            order = rng.permutation(len(targets))
            epoch_loss = 0.0
            for start in range(0, len(order), batch_size):
                selected = order[start : start + batch_size]
                loss = self._train_batch(contexts[selected], targets[selected], learning_rate)
                epoch_loss += loss * len(selected)
                self.steps += 1
            self.tokens_seen += len(targets)
            losses.append(epoch_loss / len(targets))
        return losses

    def _train_batch(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        scores, hidden, flat, cache = self._forward(x)
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        rows = np.arange(len(y))
        loss = -np.log(np.maximum(probabilities[rows, y], 1e-9)).mean()

        grad_scores = probabilities
        grad_scores[rows, y] -= 1
        grad_scores /= len(y)
        grad_w_output = self.backend.matmul(hidden.T, grad_scores)
        grad_b_output = grad_scores.sum(axis=0)
        grad_hidden = self.backend.matmul(grad_scores, self.w_output.T)
        grad_activation = grad_hidden * cache["gate"]
        grad_gate = grad_hidden * cache["activated"]
        grad_projection = grad_activation * cache["activation_derivative"]
        grad_gate_pre = grad_gate * cache["gate"] * (1.0 - cache["gate"])
        grad_w_hidden = self.backend.matmul(flat.T, grad_projection)
        grad_b_hidden = grad_projection.sum(axis=0)
        grad_w_gate = self.backend.matmul(flat.T, grad_gate_pre)
        grad_b_gate = grad_gate_pre.sum(axis=0)
        grad_w_residual = self.backend.matmul(cache["residual_input"].T, grad_hidden)
        grad_flat = self.backend.matmul(grad_projection, self.w_hidden.T)
        grad_flat += self.backend.matmul(grad_gate_pre, self.w_gate.T)
        grad_residual_input = self.backend.matmul(grad_hidden, self.w_residual.T)
        grad_embedded = grad_flat.reshape(len(y), self.shape.context_length, self.shape.embedding_size)
        grad_embedded += grad_residual_input[:, None, :] / self.shape.context_length
        grad_embedding = np.zeros_like(self.embedding)
        np.add.at(grad_embedding, x.reshape(-1), grad_embedded.reshape(-1, self.shape.embedding_size))

        gradients = [
            grad_embedding,
            grad_w_hidden,
            grad_b_hidden,
            grad_w_gate,
            grad_b_gate,
            grad_w_residual,
            grad_w_output,
            grad_b_output,
        ]
        norm = math.sqrt(sum(float(np.sum(gradient * gradient)) for gradient in gradients))
        scale = min(1.0, 5.0 / max(norm, 1e-8))
        self.embedding -= lr * grad_embedding * scale
        self.w_hidden -= lr * grad_w_hidden * scale
        self.b_hidden -= lr * grad_b_hidden * scale
        self.w_gate -= lr * grad_w_gate * scale
        self.b_gate -= lr * grad_b_gate * scale
        self.w_residual -= lr * grad_w_residual * scale
        self.w_output -= lr * grad_w_output * scale
        self.b_output -= lr * grad_b_output * scale
        return float(loss)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 160,
        temperature: float = 0.8,
        top_k: int = 32,
        seed: int | None = None,
        stop: Iterable[str] = ("\nUser:", "\n使用者：", "\nユーザー："),
    ) -> str:
        tokenizer = ByteTokenizer()
        prompt_tokens = tokenizer.encode(prompt, bos=True)
        generated: list[int] = []
        rng = np.random.default_rng(seed)
        for _ in range(max_new_tokens):
            context = ([ByteTokenizer.PAD] * self.shape.context_length + prompt_tokens + generated)[
                -self.shape.context_length :
            ]
            scores, _, _ = self.logits(np.asarray([context], dtype=np.int32), inference=True)
            logits = scores[0].astype(np.float64) / max(temperature, 0.05)
            logits[[ByteTokenizer.PAD, ByteTokenizer.BOS]] = -np.inf
            if generated:
                logits[generated[-1]] -= 0.35
            k = max(1, min(top_k, len(logits)))
            choices = np.argpartition(logits, -k)[-k:]
            choice_logits = logits[choices]
            choice_logits -= np.max(choice_logits)
            probability = np.exp(choice_logits)
            probability /= probability.sum()
            token = int(rng.choice(choices, p=probability))
            if token == ByteTokenizer.EOS:
                break
            generated.append(token)
            text = tokenizer.decode(generated)
            if any(marker in text for marker in stop):
                break
        return tokenizer.decode(generated).strip()

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": self.FORMAT_VERSION,
            "shape": asdict(self.shape),
            "steps": self.steps,
            "tokens_seen": self.tokens_seen,
        }
        np.savez_compressed(path, metadata=json.dumps(metadata), **self.parameters())
        return path

    @classmethod
    def load(cls, path: Path) -> NeuralTextModel:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            format_version = int(metadata.get("format_version", 1))
            if format_version not in cls.SUPPORTED_FORMATS:
                raise ValueError("Unsupported Momo-LM checkpoint format")
            model = cls(ModelShape(**metadata["shape"]))
            for name in model.parameters():
                if name in archive.files:
                    setattr(model, name, archive[name].astype(np.float32))
            model.steps = int(metadata.get("steps", 0))
            model.tokens_seen = int(metadata.get("tokens_seen", 0))
            return model

    def inspect(self) -> dict[str, object]:
        layers = {}
        for name, value in self.parameters().items():
            layers[name] = {
                "shape": list(value.shape),
                "parameters": int(value.size),
                "mean": float(value.mean()),
                "std": float(value.std()),
                "minimum": float(value.min()),
                "maximum": float(value.max()),
                "zeros_percent": float(np.mean(value == 0) * 100),
            }
        return {
            "architecture": "gated-mixed-neuron-language-model",
            "format_version": self.FORMAT_VERSION,
            "parameters": self.parameter_count,
            "training_steps": self.steps,
            "tokens_seen": self.tokens_seen,
            "shape": asdict(self.shape),
            "backend": self.backend.describe(),
            "layers": layers,
        }
