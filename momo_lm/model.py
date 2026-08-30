from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import os
import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .backend import TensorBackend, get_backend
from .tokenizer import ByteTokenizer


@dataclass(frozen=True, slots=True)
class CheckpointLimits:
    """Resource limits applied before any checkpoint tensor is allocated."""

    max_compressed_bytes: int = 256 * 1024 * 1024
    max_uncompressed_bytes: int = 512 * 1024 * 1024
    max_metadata_bytes: int = 1024 * 1024
    max_archive_members: int = 64


@dataclass(slots=True)
class ModelShape:
    vocab_size: int = ByteTokenizer.vocab_size
    context_length: int = 128
    embedding_size: int = 64
    hidden_size: int = 256
    attention_heads: int = 4
    neuron_groups: int = 8
    seed: int = 20260803

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.vocab_size != ByteTokenizer.vocab_size:
            raise ValueError(f"vocab_size must be {ByteTokenizer.vocab_size}")
        if not 1 <= self.context_length <= 4096:
            raise ValueError("context_length must be between 1 and 4096")
        if not 4 <= self.embedding_size <= 4096:
            raise ValueError("embedding_size must be between 4 and 4096")
        if not 4 <= self.hidden_size <= 16384:
            raise ValueError("hidden_size must be between 4 and 16384")
        if not 1 <= self.attention_heads <= self.embedding_size:
            raise ValueError("attention_heads must not exceed embedding_size")
        if self.embedding_size % self.attention_heads:
            raise ValueError("embedding_size must be divisible by attention_heads")
        if not 1 <= self.neuron_groups <= self.hidden_size:
            raise ValueError("neuron_groups must not exceed hidden_size")
        e, c, h, v, groups = (
            self.embedding_size,
            self.context_length,
            self.hidden_size,
            self.vocab_size,
            self.neuron_groups,
        )
        parameter_elements = (
            v * e
            + c * e
            + 4 * e * e
            + 2 * e
            + 3 * e * h
            + 2 * h
            + e * groups
            + 3 * groups
            + h * h
            + h
            + h * v
            + v
        )
        if parameter_elements * 4 > CheckpointLimits().max_uncompressed_bytes:
            raise ValueError("model shape exceeds the 512 MiB parameter limit")


class NeuralTextModel:
    """Compact byte language model with inspectable attention and neuron routing.

    Version 3 replaces the old flattened context projection with multi-head
    attention pooling. The pooled representation feeds routed, gated mixed
    activation groups and a residual mixing layer. Training remains deliberately
    explicit NumPy so every gradient is inspectable; matrix-heavy inference uses
    the selected native backend when one is available.
    """

    FORMAT_VERSION = 3
    SUPPORTED_FORMATS = frozenset({1, 2, 3})
    CHECKPOINT_LIMITS = CheckpointLimits()
    _FORMAT3_METADATA_KEYS = frozenset(
        {"format_version", "shape", "steps", "tokens_seen", "tensor_manifest", "training"}
    )

    def __init__(self, shape: ModelShape | None = None) -> None:
        self.shape = shape or ModelShape()
        rng = np.random.default_rng(self.shape.seed)
        e, c, h, v, groups = (
            self.shape.embedding_size,
            self.shape.context_length,
            self.shape.hidden_size,
            self.shape.vocab_size,
            self.shape.neuron_groups,
        )

        self.embedding = rng.normal(0.0, 0.055, (v, e)).astype(np.float32)
        self.position_embedding = rng.normal(0.0, 0.012, (c, e)).astype(np.float32)
        attention_scale = math.sqrt(1.0 / e)
        self.w_query = rng.normal(0.0, attention_scale, (e, e)).astype(np.float32)
        self.w_key = rng.normal(0.0, attention_scale, (e, e)).astype(np.float32)
        self.w_value = rng.normal(0.0, attention_scale, (e, e)).astype(np.float32)
        self.w_attention = rng.normal(0.0, attention_scale, (e, e)).astype(np.float32)
        self.rms_weight = np.ones(e, dtype=np.float32)
        self.pool_query = rng.normal(0.0, 0.02, e).astype(np.float32)

        hidden_scale = math.sqrt(2.0 / e)
        self.w_hidden = rng.normal(0.0, hidden_scale, (e, h)).astype(np.float32)
        self.b_hidden = np.zeros(h, dtype=np.float32)
        self.w_gate = rng.normal(0.0, 0.02, (e, h)).astype(np.float32)
        self.b_gate = np.full(h, 1.5, dtype=np.float32)
        self.w_residual = rng.normal(0.0, math.sqrt(1.0 / e), (e, h)).astype(np.float32)
        self.w_router = rng.normal(0.0, 0.02, (e, groups)).astype(np.float32)
        self.b_router = np.zeros(groups, dtype=np.float32)
        self.group_scale = np.full(groups, 0.1, dtype=np.float32)
        self.group_bias = np.zeros(groups, dtype=np.float32)
        self.w_neuron_mix = rng.normal(0.0, 0.025, (h, h)).astype(np.float32)
        self.b_neuron_mix = np.zeros(h, dtype=np.float32)
        self.w_output = rng.normal(0.0, math.sqrt(1.0 / h), (h, v)).astype(np.float32)
        self.b_output = np.zeros(v, dtype=np.float32)

        self.steps = 0
        self.tokens_seen = 0
        self.training_metadata: dict[str, Any] = {}
        self.backend: TensorBackend = get_backend()

    @property
    def parameter_count(self) -> int:
        return sum(value.size for value in self.parameters().values())

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "embedding": self.embedding,
            "position_embedding": self.position_embedding,
            "w_query": self.w_query,
            "w_key": self.w_key,
            "w_value": self.w_value,
            "w_attention": self.w_attention,
            "rms_weight": self.rms_weight,
            "pool_query": self.pool_query,
            "w_hidden": self.w_hidden,
            "b_hidden": self.b_hidden,
            "w_gate": self.w_gate,
            "b_gate": self.b_gate,
            "w_residual": self.w_residual,
            "w_router": self.w_router,
            "b_router": self.b_router,
            "group_scale": self.group_scale,
            "group_bias": self.group_bias,
            "w_neuron_mix": self.w_neuron_mix,
            "b_neuron_mix": self.b_neuron_mix,
            "w_output": self.w_output,
            "b_output": self.b_output,
        }

    def _contexts(self, tokens: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        stream, targets = self._context_stream(tokens)
        positions = np.arange(len(targets), dtype=np.int64)
        return self._context_batch(stream, targets, positions)

    def _context_stream(self, tokens: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(tokens, dtype=np.int32)
        stream = np.concatenate(
            (
                np.full(self.shape.context_length, ByteTokenizer.PAD, dtype=np.int32),
                targets,
            )
        )
        return stream, targets

    def _context_batch(
        self, stream: np.ndarray, targets: np.ndarray, positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(positions, dtype=np.int64)
        offsets = positions[:, None] + np.arange(self.shape.context_length, dtype=np.int64)
        return np.asarray(stream[offsets], dtype=np.int32), targets[positions]

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
        coefficient = math.sqrt(2.0 / math.pi)
        group_ids = self._hidden_group_ids()
        for group in range(self.shape.neuron_groups):
            columns = group_ids == group
            chunk = value[:, columns]
            if group % 3 == 0:
                output = np.tanh(chunk)
                activated[:, columns] = output
                derivative[:, columns] = 1.0 - output * output
            elif group % 3 == 1:
                transformed = coefficient * (chunk + 0.044715 * chunk**3)
                tangent = np.tanh(transformed)
                activated[:, columns] = 0.5 * chunk * (1.0 + tangent)
                derivative[:, columns] = 0.5 * (1.0 + tangent) + 0.5 * chunk * (
                    1.0 - tangent * tangent
                ) * coefficient * (1.0 + 3.0 * 0.044715 * chunk * chunk)
            else:
                sigmoid = self._sigmoid(chunk)
                activated[:, columns] = chunk * sigmoid
                derivative[:, columns] = sigmoid + chunk * sigmoid * (1.0 - sigmoid)
        return activated, derivative

    def _hidden_group_ids(self) -> np.ndarray:
        indices = np.arange(self.shape.hidden_size, dtype=np.int32)
        return np.minimum(
            indices * self.shape.neuron_groups // self.shape.hidden_size,
            self.shape.neuron_groups - 1,
        )

    @staticmethod
    def _softmax(value: np.ndarray, axis: int = -1) -> np.ndarray:
        shifted = value - np.max(value, axis=axis, keepdims=True)
        exponential = np.exp(shifted)
        return exponential / np.sum(exponential, axis=axis, keepdims=True)

    def _matmul(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        original_shape = left.shape[:-1]
        flat = np.ascontiguousarray(left.reshape(-1, left.shape[-1]), dtype=np.float32)
        result = self.backend.matmul(flat, right)
        return result.reshape(*original_shape, right.shape[1])

    def _forward(
        self, contexts: np.ndarray, *, inference: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        contexts = np.asarray(contexts, dtype=np.int32)
        if contexts.ndim != 2 or contexts.shape[1:] != (self.shape.context_length,):
            raise ValueError(f"contexts must have shape (batch, {self.shape.context_length})")
        if contexts.size and (contexts.min() < 0 or contexts.max() >= self.shape.vocab_size):
            raise ValueError("context token is outside the vocabulary")

        embedded = self.embedding[contexts] + self.position_embedding[None, :, :]
        mean_square = np.mean(embedded * embedded, axis=-1, keepdims=True)
        inverse_rms = 1.0 / np.sqrt(mean_square + 1e-5)
        normalized_base = embedded * inverse_rms
        if inference and contexts.shape[0] <= 8:
            normalized = self.backend.rms_norm(
                embedded.reshape(-1, self.shape.embedding_size), self.rms_weight
            ).reshape(embedded.shape)
        else:
            normalized = normalized_base * self.rms_weight

        batch = contexts.shape[0]
        heads = self.shape.attention_heads
        head_size = self.shape.embedding_size // heads
        query = self._matmul(normalized, self.w_query).reshape(
            batch, self.shape.context_length, heads, head_size
        )
        key = self._matmul(normalized, self.w_key).reshape(
            batch, self.shape.context_length, heads, head_size
        )
        value = self._matmul(normalized, self.w_value).reshape(
            batch, self.shape.context_length, heads, head_size
        )
        pooled_query = query[:, -1] + self.pool_query.reshape(heads, head_size)
        attention_scores = np.einsum("bhd,bchd->bhc", pooled_query, key, optimize=True)
        attention_scores *= 1.0 / math.sqrt(head_size)
        valid = contexts != ByteTokenizer.PAD
        valid[:, -1] = True
        if inference and batch <= 8:
            attention_heads = np.empty((batch, heads, head_size), dtype=np.float32)
            for row in range(batch):
                first = int(np.flatnonzero(valid[row])[0])
                native_query = np.ascontiguousarray(query[row, first:]).copy()
                native_query[-1] += self.pool_query.reshape(heads, head_size)
                native_attention = self.backend.causal_gqa(
                    native_query,
                    np.ascontiguousarray(key[row, first:]),
                    np.ascontiguousarray(value[row, first:]),
                    scale=1.0 / math.sqrt(head_size),
                )
                attention_heads[row] = native_attention[-1]
            attention_probability = np.empty((0,), dtype=np.float32)
        else:
            attention_scores = np.where(valid[:, None, :], attention_scores, -1.0e9)
            attention_probability = self._softmax(attention_scores, axis=-1)
            attention_heads = np.einsum(
                "bhc,bchd->bhd", attention_probability, value, optimize=True
            )
        attention_concat = attention_heads.reshape(batch, self.shape.embedding_size)
        pooled = self.backend.matmul(attention_concat, self.w_attention) + normalized[:, -1]

        projection = self.backend.matmul(pooled, self.w_hidden) + self.b_hidden
        gate = self._sigmoid(self.backend.matmul(pooled, self.w_gate) + self.b_gate)
        activated, activation_derivative = self._activate(projection)
        router = self._softmax(
            self.backend.matmul(pooled, self.w_router) + self.b_router, axis=-1
        )
        group_ids = self._hidden_group_ids()
        route_factor = (
            1.0
            + router[:, group_ids] * self.group_scale[group_ids]
            + self.group_bias[group_ids]
        )
        base = activated * gate * route_factor + self.backend.matmul(pooled, self.w_residual)
        mixed = np.tanh(self.backend.matmul(base, self.w_neuron_mix) + self.b_neuron_mix)
        hidden = base + mixed
        scores = self.backend.matmul(hidden, self.w_output) + self.b_output

        if inference:
            return scores, hidden, pooled, {}
        cache = {
            "contexts": contexts,
            "embedded": embedded,
            "inverse_rms": inverse_rms,
            "normalized_base": normalized_base,
            "normalized": normalized,
            "query": query,
            "key": key,
            "value": value,
            "pooled_query": pooled_query,
            "attention_probability": attention_probability,
            "attention_concat": attention_concat,
            "pooled": pooled,
            "gate": gate,
            "activated": activated,
            "activation_derivative": activation_derivative,
            "router": router,
            "route_factor": route_factor,
            "base": base,
            "mixed": mixed,
        }
        return scores, hidden, pooled, cache

    def logits(
        self, contexts: np.ndarray, *, inference: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, hidden, pooled, _ = self._forward(contexts, inference=inference)
        return scores, hidden, pooled

    def _loss_and_gradients(
        self, contexts: np.ndarray, targets: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        scores, hidden, _, cache = self._forward(contexts)
        shifted = scores - scores.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        rows = np.arange(len(targets))
        loss = -np.log(np.maximum(probabilities[rows, targets], 1e-12)).mean()

        grad_scores = probabilities
        grad_scores[rows, targets] -= 1.0
        grad_scores /= len(targets)
        gradients = {name: np.zeros_like(value) for name, value in self.parameters().items()}
        gradients["w_output"] = self.backend.matmul(hidden.T, grad_scores)
        gradients["b_output"] = grad_scores.sum(axis=0)
        grad_hidden = self.backend.matmul(grad_scores, self.w_output.T)

        grad_mix_pre = grad_hidden * (1.0 - cache["mixed"] ** 2)
        gradients["w_neuron_mix"] = self.backend.matmul(cache["base"].T, grad_mix_pre)
        gradients["b_neuron_mix"] = grad_mix_pre.sum(axis=0)
        grad_base = grad_hidden + self.backend.matmul(grad_mix_pre, self.w_neuron_mix.T)

        group_ids = self._hidden_group_ids()
        activation_gate = cache["activated"] * cache["gate"]
        grad_activated = grad_base * cache["gate"] * cache["route_factor"]
        grad_gate = grad_base * cache["activated"] * cache["route_factor"]
        grad_route_factor = grad_base * activation_gate
        grad_projection = grad_activated * cache["activation_derivative"]
        grad_gate_pre = grad_gate * cache["gate"] * (1.0 - cache["gate"])

        grad_router = np.zeros_like(cache["router"])
        for group in range(self.shape.neuron_groups):
            columns = group_ids == group
            group_route_gradient = grad_route_factor[:, columns].sum(axis=1)
            gradients["group_scale"][group] = np.sum(
                group_route_gradient * cache["router"][:, group]
            )
            gradients["group_bias"][group] = np.sum(group_route_gradient)
            grad_router[:, group] = group_route_gradient * self.group_scale[group]
        grad_router_pre = cache["router"] * (
            grad_router - np.sum(grad_router * cache["router"], axis=1, keepdims=True)
        )

        pooled = cache["pooled"]
        gradients["w_hidden"] = self.backend.matmul(pooled.T, grad_projection)
        gradients["b_hidden"] = grad_projection.sum(axis=0)
        gradients["w_gate"] = self.backend.matmul(pooled.T, grad_gate_pre)
        gradients["b_gate"] = grad_gate_pre.sum(axis=0)
        gradients["w_residual"] = self.backend.matmul(pooled.T, grad_base)
        gradients["w_router"] = self.backend.matmul(pooled.T, grad_router_pre)
        gradients["b_router"] = grad_router_pre.sum(axis=0)
        grad_pooled = self.backend.matmul(grad_projection, self.w_hidden.T)
        grad_pooled += self.backend.matmul(grad_gate_pre, self.w_gate.T)
        grad_pooled += self.backend.matmul(grad_base, self.w_residual.T)
        grad_pooled += self.backend.matmul(grad_router_pre, self.w_router.T)

        gradients["w_attention"] = self.backend.matmul(
            cache["attention_concat"].T, grad_pooled
        )
        grad_attention_concat = self.backend.matmul(grad_pooled, self.w_attention.T)
        batch = len(targets)
        heads = self.shape.attention_heads
        head_size = self.shape.embedding_size // heads
        grad_attention_heads = grad_attention_concat.reshape(batch, heads, head_size)
        grad_probability = np.einsum(
            "bhd,bchd->bhc", grad_attention_heads, cache["value"], optimize=True
        )
        grad_value = np.einsum(
            "bhc,bhd->bchd",
            cache["attention_probability"],
            grad_attention_heads,
            optimize=True,
        )
        grad_attention_scores = cache["attention_probability"] * (
            grad_probability
            - np.sum(
                grad_probability * cache["attention_probability"], axis=-1, keepdims=True
            )
        )
        valid = cache["contexts"] != ByteTokenizer.PAD
        valid[:, -1] = True
        grad_attention_scores *= valid[:, None, :]
        attention_scale = 1.0 / math.sqrt(head_size)
        grad_pooled_query = np.einsum(
            "bhc,bchd->bhd", grad_attention_scores, cache["key"], optimize=True
        ) * attention_scale
        grad_key = np.einsum(
            "bhc,bhd->bchd", grad_attention_scores, cache["pooled_query"], optimize=True
        ) * attention_scale
        grad_query = np.zeros_like(cache["query"])
        grad_query[:, -1] = grad_pooled_query
        gradients["pool_query"] = grad_pooled_query.sum(axis=0).reshape(-1)

        grad_query_flat = grad_query.reshape(-1, self.shape.embedding_size)
        grad_key_flat = grad_key.reshape(-1, self.shape.embedding_size)
        grad_value_flat = grad_value.reshape(-1, self.shape.embedding_size)
        normalized_flat = cache["normalized"].reshape(-1, self.shape.embedding_size)
        gradients["w_query"] = self.backend.matmul(normalized_flat.T, grad_query_flat)
        gradients["w_key"] = self.backend.matmul(normalized_flat.T, grad_key_flat)
        gradients["w_value"] = self.backend.matmul(normalized_flat.T, grad_value_flat)
        grad_normalized = self.backend.matmul(grad_query_flat, self.w_query.T)
        grad_normalized += self.backend.matmul(grad_key_flat, self.w_key.T)
        grad_normalized += self.backend.matmul(grad_value_flat, self.w_value.T)
        grad_normalized = grad_normalized.reshape(cache["normalized"].shape)
        grad_normalized[:, -1] += grad_pooled

        gradients["rms_weight"] = np.sum(
            grad_normalized * cache["normalized_base"], axis=(0, 1)
        )
        grad_normalized_base = grad_normalized * self.rms_weight
        grad_embedded = cache["inverse_rms"] * (
            grad_normalized_base
            - cache["normalized_base"]
            * np.mean(
                grad_normalized_base * cache["normalized_base"], axis=-1, keepdims=True
            )
        )
        gradients["position_embedding"] = grad_embedded.sum(axis=0)
        np.add.at(
            gradients["embedding"],
            cache["contexts"].reshape(-1),
            grad_embedded.reshape(-1, self.shape.embedding_size),
        )
        return float(loss), gradients

    def train_text(
        self,
        text: str,
        *,
        epochs: int = 1,
        learning_rate: float = 0.002,
        batch_size: int = 144,
        seed: int | None = None,
        replay_texts: Sequence[str] = (),
        validation_text: str | None = None,
        validation_is_held_out: bool = False,
    ) -> list[float]:
        """Train with deterministic AdamW and return one mean loss per epoch."""

        from .training import AdamWConfig, AdamWTrainer

        if isinstance(epochs, bool) or not isinstance(epochs, int):
            raise TypeError("epochs must be an integer")
        trainer = AdamWTrainer(
            self,
            AdamWConfig(
                epochs=max(1, epochs),
                learning_rate=learning_rate,
                batch_size=batch_size,
                seed=self.shape.seed if seed is None else seed,
            ),
        )
        result = trainer.fit(
            text,
            replay_texts=replay_texts,
            validation_text=validation_text,
            validation_is_held_out=validation_is_held_out,
        )
        return result.epoch_losses

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
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or not 0 <= max_new_tokens <= 4096
        ):
            raise ValueError("max_new_tokens must be an integer in [0, 4096]")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise TypeError("temperature must be numeric")
        if not math.isfinite(float(temperature)) or not 0.0 < float(temperature) <= 10.0:
            raise ValueError("temperature must be finite and in (0, 10]")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 259:
            raise ValueError("top_k must be an integer in [1, 259]")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64
        ):
            raise ValueError("seed must be None or an integer fitting uint64")
        tokenizer = ByteTokenizer()
        prompt_tokens = tokenizer.encode(prompt, bos=True)
        generated: list[int] = []
        rng = np.random.default_rng(seed)
        for _ in range(max_new_tokens):
            context = (
                [ByteTokenizer.PAD] * self.shape.context_length + prompt_tokens + generated
            )[-self.shape.context_length :]
            scores, _, _ = self.logits(np.asarray([context], dtype=np.int32), inference=True)
            logits = scores[0].astype(np.float64) / max(float(temperature), 0.05)
            logits[[ByteTokenizer.PAD, ByteTokenizer.BOS]] = -np.inf
            if generated:
                logits[generated[-1]] -= 0.35
            k = min(top_k, len(logits))
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

    @staticmethod
    def _tensor_digest(value: np.ndarray) -> str:
        canonical = np.ascontiguousarray(value, dtype="<f4")
        return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()

    def _manifest(self) -> dict[str, dict[str, Any]]:
        manifest: dict[str, dict[str, Any]] = {}
        for name, value in self.parameters().items():
            canonical = np.ascontiguousarray(value, dtype="<f4")
            manifest[name] = {
                "shape": list(canonical.shape),
                "dtype": "float32-le",
                "nbytes": int(canonical.nbytes),
                "sha256": self._tensor_digest(canonical),
            }
        return manifest

    def _metadata(self) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "shape": asdict(self.shape),
            "steps": int(self.steps),
            "tokens_seen": int(self.tokens_seen),
            "tensor_manifest": self._manifest(),
            "training": dict(self.training_metadata),
        }

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _copy_atomic(cls, source: Path, target: Path) -> None:
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
            cls._fsync_directory(target.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        backup = path.with_suffix(path.suffix + ".last-good")
        try:
            metadata = json.dumps(
                self._metadata(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            tensors = {
                name: np.ascontiguousarray(value, dtype="<f4")
                for name, value in self.parameters().items()
            }
            with os.fdopen(descriptor, "wb") as stream:
                self._write_deterministic_npz(
                    stream, {"metadata": np.asarray(metadata), **tensors}
                )
                stream.flush()
                os.fsync(stream.fileno())
            self._load_checkpoint(temporary)
            if path.exists():
                try:
                    self._load_checkpoint(path)
                except (OSError, ValueError, zipfile.BadZipFile):
                    if not backup.exists():
                        self._copy_atomic(temporary, backup)
                else:
                    self._copy_atomic(path, backup)
            elif not backup.exists():
                self._copy_atomic(temporary, backup)
            os.replace(temporary, path)
            self._fsync_directory(path.parent)
            return path
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _write_deterministic_npz(stream: Any, arrays: Mapping[str, np.ndarray]) -> None:
        """Write stable NPZ bytes so identical weights have an identical file hash."""

        with zipfile.ZipFile(
            stream, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name, value in arrays.items():
                payload = io.BytesIO()
                np.save(payload, value, allow_pickle=False)
                member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                member.external_attr = 0o600 << 16
                archive.writestr(member, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED)

    @classmethod
    def _check_archive_limits(cls, path: Path) -> None:
        limits = cls.CHECKPOINT_LIMITS
        try:
            compressed_size = path.stat().st_size
        except OSError as exc:
            raise ValueError(f"cannot read Momo-LM checkpoint: {exc}") from exc
        if compressed_size > limits.max_compressed_bytes:
            raise ValueError("checkpoint exceeds the 256 MiB compressed-size limit")
        try:
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                if not 1 <= len(members) <= limits.max_archive_members:
                    raise ValueError("checkpoint has an invalid number of archive members")
                names = [member.filename for member in members]
                if len(names) != len(set(names)):
                    raise ValueError("checkpoint contains duplicate archive members")
                if any(
                    member.flag_bits & 0x1
                    or "/" in member.filename
                    or "\\" in member.filename
                    or not member.filename.endswith(".npy")
                    for member in members
                ):
                    raise ValueError("checkpoint contains an unsafe archive member")
                total = sum(member.file_size for member in members)
                if total > limits.max_uncompressed_bytes:
                    raise ValueError("checkpoint exceeds the 512 MiB unpacked-size limit")
                metadata = next(
                    (member for member in members if member.filename == "metadata.npy"), None
                )
                if metadata is None or metadata.file_size > limits.max_metadata_bytes:
                    raise ValueError("checkpoint metadata is missing or too large")
        except zipfile.BadZipFile as exc:
            raise ValueError("checkpoint is not a valid NPZ archive") from exc

    @classmethod
    def _parse_metadata(cls, archive: Mapping[str, np.ndarray]) -> dict[str, Any]:
        if "metadata" not in archive:
            raise ValueError("checkpoint metadata is missing")
        value = archive["metadata"]
        if value.shape != () or value.dtype.kind not in {"U", "S"}:
            raise ValueError("checkpoint metadata has an invalid representation")
        item = value.item()
        if isinstance(item, bytes):
            try:
                encoded = item.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("checkpoint metadata is not valid UTF-8") from exc
        else:
            encoded = str(item)
        if len(encoded.encode("utf-8")) > cls.CHECKPOINT_LIMITS.max_metadata_bytes:
            raise ValueError("checkpoint metadata is too large")

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item_value in pairs:
                if key in result:
                    raise ValueError(f"duplicate checkpoint metadata key: {key}")
                result[key] = item_value
            return result

        try:
            result = json.loads(encoded, object_pairs_hook=unique_object)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("checkpoint metadata is not valid JSON") from exc
        if not isinstance(result, dict):
            raise ValueError("checkpoint metadata must be an object")
        return result

    @classmethod
    def _validate_counter(cls, value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
            raise ValueError(f"checkpoint {name} is invalid")
        return value

    @classmethod
    def _shape_from_format3(cls, value: Any) -> ModelShape:
        expected = frozenset(asdict(ModelShape()).keys())
        if not isinstance(value, dict) or frozenset(value) != expected:
            raise ValueError("format 3 checkpoint shape keys are invalid")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value.values()):
            raise ValueError("format 3 checkpoint shape values must be integers")
        try:
            return ModelShape(**value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"format 3 checkpoint shape is invalid: {exc}") from exc

    @classmethod
    def _load_format3(
        cls, archive: Mapping[str, np.ndarray], files: set[str], metadata: dict[str, Any]
    ) -> NeuralTextModel:
        if frozenset(metadata) != cls._FORMAT3_METADATA_KEYS:
            raise ValueError("format 3 checkpoint metadata keys are invalid")
        shape = cls._shape_from_format3(metadata["shape"])
        model = cls(shape)
        expected_tensors = model.parameters()
        expected_files = {"metadata", *expected_tensors}
        if files != expected_files:
            raise ValueError("format 3 checkpoint tensors are missing or unexpected")
        manifest = metadata["tensor_manifest"]
        if not isinstance(manifest, dict) or set(manifest) != set(expected_tensors):
            raise ValueError("format 3 tensor manifest is incomplete")
        total_bytes = 0
        for name, destination in expected_tensors.items():
            entry = manifest[name]
            expected_entry_keys = {"shape", "dtype", "nbytes", "sha256"}
            if not isinstance(entry, dict) or set(entry) != expected_entry_keys:
                raise ValueError(f"tensor manifest entry {name!r} is invalid")
            value = archive[name]
            if value.dtype != np.dtype("<f4") or value.shape != destination.shape:
                raise ValueError(f"checkpoint tensor {name!r} has an invalid dtype or shape")
            if (
                entry["shape"] != list(destination.shape)
                or entry["dtype"] != "float32-le"
                or isinstance(entry["nbytes"], bool)
                or entry["nbytes"] != value.nbytes
                or not isinstance(entry["sha256"], str)
            ):
                raise ValueError(f"tensor manifest entry {name!r} does not match the tensor")
            total_bytes += value.nbytes
            if total_bytes > cls.CHECKPOINT_LIMITS.max_uncompressed_bytes:
                raise ValueError("checkpoint tensors exceed the unpacked-size limit")
            digest = cls._tensor_digest(value)
            if not hmac.compare_digest(digest, entry["sha256"]):
                raise ValueError(f"checkpoint tensor {name!r} failed SHA-256 verification")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"checkpoint tensor {name!r} contains non-finite values")
            setattr(model, name, np.array(value, dtype=np.float32, order="C", copy=True))
        model.steps = cls._validate_counter(metadata["steps"], "steps")
        model.tokens_seen = cls._validate_counter(metadata["tokens_seen"], "tokens_seen")
        training = metadata["training"]
        if not isinstance(training, dict):
            raise ValueError("checkpoint training metadata must be an object")
        try:
            json.dumps(training, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint training metadata is not JSON-safe") from exc
        model.training_metadata = dict(training)
        return model

    @classmethod
    def _legacy_shape(cls, value: Any, version: int) -> tuple[ModelShape, int]:
        if not isinstance(value, dict):
            raise ValueError("legacy checkpoint shape is invalid")
        required = {"vocab_size", "context_length", "embedding_size", "hidden_size", "seed"}
        allowed = required | ({"neuron_group_size"} if version == 2 else set())
        if not required.issubset(value) or not set(value).issubset(allowed):
            raise ValueError("legacy checkpoint shape keys are invalid")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value.values()):
            raise ValueError("legacy checkpoint shape values must be integers")
        embedding_size = value["embedding_size"]
        heads = min(4, embedding_size)
        while embedding_size % heads:
            heads -= 1
        group_size = int(value.get("neuron_group_size", max(1, value["hidden_size"] // 8)))
        if group_size <= 0:
            raise ValueError("legacy neuron_group_size is invalid")
        groups = min(value["hidden_size"], math.ceil(value["hidden_size"] / group_size))
        shape = ModelShape(
            vocab_size=value["vocab_size"],
            context_length=value["context_length"],
            embedding_size=embedding_size,
            hidden_size=value["hidden_size"],
            attention_heads=heads,
            neuron_groups=groups,
            seed=value["seed"],
        )
        return shape, group_size

    @classmethod
    def _load_legacy(
        cls,
        archive: Mapping[str, np.ndarray],
        files: set[str],
        metadata: dict[str, Any],
        version: int,
    ) -> NeuralTextModel:
        allowed_metadata = {"format_version", "shape", "steps", "tokens_seen"}
        if not {"shape"}.issubset(metadata) or not set(metadata).issubset(allowed_metadata):
            raise ValueError("legacy checkpoint metadata keys are invalid")
        shape, group_size = cls._legacy_shape(metadata["shape"], version)
        model = cls(shape)
        common = {"embedding", "w_hidden", "b_hidden", "w_output", "b_output"}
        expected = common if version == 1 else common | {"w_gate", "b_gate", "w_residual"}
        if files != {"metadata", *expected}:
            raise ValueError(f"format {version} checkpoint tensors are missing or unexpected")
        legacy_shapes = {
            "embedding": (shape.vocab_size, shape.embedding_size),
            "w_hidden": (shape.context_length * shape.embedding_size, shape.hidden_size),
            "b_hidden": (shape.hidden_size,),
            "w_output": (shape.hidden_size, shape.vocab_size),
            "b_output": (shape.vocab_size,),
            "w_gate": (shape.context_length * shape.embedding_size, shape.hidden_size),
            "b_gate": (shape.hidden_size,),
            "w_residual": (shape.embedding_size, shape.hidden_size),
        }
        for name in expected:
            value = archive[name]
            if value.dtype.kind != "f" or value.shape != legacy_shapes[name]:
                raise ValueError(f"legacy checkpoint tensor {name!r} has an invalid dtype or shape")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"legacy checkpoint tensor {name!r} contains non-finite values")
        model.embedding = np.asarray(archive["embedding"], dtype=np.float32).copy()
        # Attention pooling produces one embedding vector. Summing the legacy
        # per-position projections is the deterministic closest mapping when a
        # flattened context matrix cannot be represented exactly by one vector.
        model.w_hidden = np.asarray(archive["w_hidden"], dtype=np.float32).reshape(
            shape.context_length, shape.embedding_size, shape.hidden_size
        ).sum(axis=0)
        model.b_hidden = np.asarray(archive["b_hidden"], dtype=np.float32).copy()
        model.w_output = np.asarray(archive["w_output"], dtype=np.float32).copy()
        model.b_output = np.asarray(archive["b_output"], dtype=np.float32).copy()
        if version == 2:
            model.w_gate = np.asarray(archive["w_gate"], dtype=np.float32).reshape(
                shape.context_length, shape.embedding_size, shape.hidden_size
            ).sum(axis=0)
            model.b_gate = np.asarray(archive["b_gate"], dtype=np.float32).copy()
            model.w_residual = np.asarray(archive["w_residual"], dtype=np.float32).copy()
        model.steps = cls._validate_counter(metadata.get("steps", 0), "steps")
        model.tokens_seen = cls._validate_counter(metadata.get("tokens_seen", 0), "tokens_seen")
        model.training_metadata = {
            "migration": {
                "source_format": version,
                "strategy": "sum-legacy-context-projections",
                "legacy_neuron_group_size": group_size,
            }
        }
        return model

    @classmethod
    def _load_checkpoint(cls, path: Path) -> NeuralTextModel:
        cls._check_archive_limits(path)
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = cls._parse_metadata(archive)
                version = metadata.get("format_version", 1)
                if isinstance(version, bool) or not isinstance(version, int):
                    raise ValueError("checkpoint format_version is invalid")
                if version not in cls.SUPPORTED_FORMATS:
                    raise ValueError("unsupported Momo-LM checkpoint format")
                files = set(archive.files)
                if version == 3:
                    return cls._load_format3(archive, files, metadata)
                return cls._load_legacy(archive, files, metadata, version)
        except (OSError, EOFError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
            raise ValueError(f"cannot load Momo-LM checkpoint: {exc}") from exc

    @classmethod
    def load(cls, path: Path, *, recover: bool = True) -> NeuralTextModel:
        path = Path(path)
        try:
            return cls._load_checkpoint(path)
        except ValueError as primary_error:
            backup = path.with_suffix(path.suffix + ".last-good")
            if not recover or not backup.is_file():
                raise
            try:
                model = cls._load_checkpoint(backup)
            except ValueError:
                raise primary_error from None
            model.training_metadata = {
                **model.training_metadata,
                "recovery": {"source": "last-good", "primary": path.name},
            }
            return model

    def inspect(self) -> dict[str, object]:
        layers: dict[str, dict[str, object]] = {}
        for name, value in self.parameters().items():
            layers[name] = {
                "shape": list(value.shape),
                "parameters": int(value.size),
                "mean": float(value.mean()),
                "std": float(value.std()),
                "minimum": float(value.min()),
                "maximum": float(value.max()),
                "zeros_percent": float(np.mean(value == 0) * 100),
                "sha256": self._tensor_digest(value),
            }
        return {
            "architecture": "attention-routed-gated-neuron-language-model",
            "format_version": self.FORMAT_VERSION,
            "parameters": self.parameter_count,
            "training_steps": self.steps,
            "tokens_seen": self.tokens_seen,
            "shape": asdict(self.shape),
            "backend": self.backend.describe(),
            "training": dict(self.training_metadata),
            "layers": layers,
        }
