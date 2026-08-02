from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .tokenizer import ByteTokenizer


@dataclass(slots=True)
class ModelShape:
    vocab_size: int = ByteTokenizer.vocab_size
    context_length: int = 24
    embedding_size: int = 32
    hidden_size: int = 96
    seed: int = 20260803


class NeuralTextModel:
    """A compact byte-level neural language model with manual NumPy backpropagation.

    The network is intentionally small and inspectable: token embedding -> ordered
    context projection -> tanh hidden layer -> next-byte distribution.
    """

    FORMAT_VERSION = 1

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
        self.w_output = rng.normal(0, math.sqrt(2 / h), (h, v)).astype(np.float32)
        self.b_output = np.zeros(v, dtype=np.float32)
        self.steps = 0
        self.tokens_seen = 0

    @property
    def parameter_count(self) -> int:
        return sum(value.size for value in self.parameters().values())

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "embedding": self.embedding,
            "w_hidden": self.w_hidden,
            "b_hidden": self.b_hidden,
            "w_output": self.w_output,
            "b_output": self.b_output,
        }

    def _contexts(self, tokens: list[int]) -> tuple[np.ndarray, np.ndarray]:
        c = self.shape.context_length
        stream = [ByteTokenizer.PAD] * c + tokens
        x = np.asarray([stream[index : index + c] for index in range(len(tokens))], dtype=np.int32)
        y = np.asarray(tokens, dtype=np.int32)
        return x, y

    def logits(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        embedded = self.embedding[contexts]
        flat = embedded.reshape(contexts.shape[0], -1)
        hidden = np.tanh(flat @ self.w_hidden + self.b_hidden)
        return hidden @ self.w_output + self.b_output, hidden, flat

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
        scores, hidden, flat = self.logits(x)
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        rows = np.arange(len(y))
        loss = -np.log(np.maximum(probabilities[rows, y], 1e-9)).mean()

        grad_scores = probabilities
        grad_scores[rows, y] -= 1
        grad_scores /= len(y)
        grad_w_output = hidden.T @ grad_scores
        grad_b_output = grad_scores.sum(axis=0)
        grad_hidden = (grad_scores @ self.w_output.T) * (1 - hidden * hidden)
        grad_w_hidden = flat.T @ grad_hidden
        grad_b_hidden = grad_hidden.sum(axis=0)
        grad_flat = grad_hidden @ self.w_hidden.T
        grad_embedded = grad_flat.reshape(len(y), self.shape.context_length, self.shape.embedding_size)
        grad_embedding = np.zeros_like(self.embedding)
        np.add.at(grad_embedding, x.reshape(-1), grad_embedded.reshape(-1, self.shape.embedding_size))

        gradients = [grad_embedding, grad_w_hidden, grad_b_hidden, grad_w_output, grad_b_output]
        norm = math.sqrt(sum(float(np.sum(gradient * gradient)) for gradient in gradients))
        scale = min(1.0, 5.0 / max(norm, 1e-8))
        self.embedding -= lr * grad_embedding * scale
        self.w_hidden -= lr * grad_w_hidden * scale
        self.b_hidden -= lr * grad_b_hidden * scale
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
            scores, _, _ = self.logits(np.asarray([context], dtype=np.int32))
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
            if metadata.get("format_version") != cls.FORMAT_VERSION:
                raise ValueError("Unsupported Momo-LM checkpoint format")
            model = cls(ModelShape(**metadata["shape"]))
            for name in model.parameters():
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
            "architecture": "ordered-context-neural-language-model",
            "format_version": self.FORMAT_VERSION,
            "parameters": self.parameter_count,
            "training_steps": self.steps,
            "tokens_seen": self.tokens_seen,
            "shape": asdict(self.shape),
            "layers": layers,
        }
