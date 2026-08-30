from __future__ import annotations

import ctypes
import math
import os
import sys
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ABI_VERSION = 2
_UINT64_MASK = (1 << 64) - 1
_FLOAT_MIN_NORMAL = np.finfo(np.float32).tiny


def _tensor(value: np.ndarray, dimensions: int, name: str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    if result.ndim != dimensions or any(size <= 0 for size in result.shape):
        raise ValueError(f"{name} must be a non-empty {dimensions}-dimensional float32 tensor")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return result


def _matrix(value: np.ndarray, name: str = "matrix") -> np.ndarray:
    return _tensor(value, 2, name)


def _vector(value: np.ndarray, name: str = "vector") -> np.ndarray:
    return _tensor(value, 1, name)


def _tensor3(value: np.ndarray, name: str) -> np.ndarray:
    return _tensor(value, 3, name)


def _positions(value: np.ndarray, tokens: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != tokens:
        raise ValueError("positions must be a one-dimensional tensor matching token count")
    validated: list[int] = []
    for item in array:
        try:
            integer = int(item)
            equal = bool(item == integer)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("positions must contain finite integers") from exc
        if not equal or not 0 <= integer <= _UINT64_MASK:
            raise ValueError("positions must contain uint64 integers")
        validated.append(integer)
    return np.ascontiguousarray(validated, dtype=np.uint64)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _mixed_activation(value: np.ndarray, group_size: int) -> np.ndarray:
    if not isinstance(group_size, (int, np.integer)) or int(group_size) <= 0:
        raise ValueError("group_size must be a positive integer")
    output = np.empty_like(value)
    for start in range(0, value.shape[1], int(group_size)):
        stop = min(start + int(group_size), value.shape[1])
        group = start // int(group_size)
        chunk = value[:, start:stop]
        if group % 3 == 0:
            output[:, start:stop] = np.tanh(chunk)
        elif group % 3 == 1:
            transformed = np.sqrt(np.float32(2.0 / np.pi)) * (
                chunk + np.float32(0.044715) * chunk**3
            )
            output[:, start:stop] = 0.5 * chunk * (1.0 + np.tanh(transformed))
        else:
            output[:, start:stop] = chunk * _sigmoid(chunk)
    return output


def _attention_shapes(
    query: np.ndarray, key: np.ndarray, value: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = _tensor3(query, "query")
    key = _tensor3(key, "key")
    value = _tensor3(value, "value")
    if key.shape != value.shape or query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
        raise ValueError("attention tensor shapes do not match")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by key/value head count")
    return query, key, value


def _decode_shapes(
    query: np.ndarray, key_cache: np.ndarray, value_cache: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = _matrix(query, "query")
    key_cache = _tensor3(key_cache, "key_cache")
    value_cache = _tensor3(value_cache, "value_cache")
    if (
        key_cache.shape != value_cache.shape
        or query.shape[1] != key_cache.shape[2]
        or query.shape[0] % key_cache.shape[1] != 0
    ):
        raise ValueError("decode attention tensor shapes do not match")
    return query, key_cache, value_cache


def _scale(value: float | None, head_size: int) -> np.float32:
    result = np.float32(1.0 / math.sqrt(head_size) if value is None else value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("attention scale must be finite and positive")
    return result


def _sampling_arguments(
    logits: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    counter: int,
) -> tuple[np.ndarray, np.float32, int, np.float32, int, int]:
    logits = _vector(logits, "logits")
    temperature32 = np.float32(temperature)
    top_p32 = np.float32(top_p)
    if not np.isfinite(temperature32) or temperature32 < 0:
        raise ValueError("temperature must be finite and non-negative")
    if not isinstance(top_k, (int, np.integer)) or top_k < 0 or top_k > logits.size:
        raise ValueError("top_k must be between zero and the logits count")
    if not np.isfinite(top_p32) or not 0 < top_p32 <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")
    for name, number in (("seed", seed), ("counter", counter)):
        if not isinstance(number, (int, np.integer)) or not 0 <= int(number) <= _UINT64_MASK:
            raise ValueError(f"{name} must fit uint64")
    return logits, temperature32, int(top_k), top_p32, int(seed), int(counter)


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def _uniform_sample(seed: int, counter: int) -> float:
    mixed = _splitmix64(
        (seed + 0x9E3779B97F4A7C15 * ((counter + 1) & _UINT64_MASK)) & _UINT64_MASK
    )
    return (mixed >> 11) * (1.0 / 9007199254740992.0)


def _rope_angle(position: int, dimension: int, rotary_dim: int, theta: float) -> float:
    frequency = math.pow(theta, -dimension / rotary_dim)
    phase = 0.0
    tau = 6.283185307179586476925286766559005768
    for shift in (48, 32, 16, 0):
        chunk = (position >> shift) & 0xFFFF
        phase = math.fmod(phase * 65536.0 + chunk * frequency, tau)
    return phase


class TensorBackend:
    name = "numpy"
    native = False

    def matmul(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left = _matrix(left, "left")
        right = _matrix(right, "right")
        if left.shape[1] != right.shape[0]:
            raise ValueError("matmul inner dimensions do not match")
        output = left @ right
        if not np.isfinite(output).all():
            raise FloatingPointError("matmul produced NaN or infinite values")
        return np.asarray(output, dtype=np.float32)

    def softmax(self, value: np.ndarray) -> np.ndarray:
        value = _matrix(value)
        shifted = value - value.max(axis=1, keepdims=True)
        exponential = np.exp(shifted)
        result = exponential / exponential.sum(axis=1, keepdims=True, dtype=np.float64)
        return np.asarray(result, dtype=np.float32)

    def layer_norm(self, value: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        value = _matrix(value)
        epsilon32 = np.float32(epsilon)
        if not np.isfinite(epsilon32) or epsilon32 <= 0:
            raise ValueError("epsilon must be finite and positive")
        values64 = value.astype(np.float64)
        mean = values64.mean(axis=1, keepdims=True)
        variance = values64.var(axis=1, keepdims=True)
        return np.asarray((values64 - mean) / np.sqrt(variance + epsilon32), dtype=np.float32)

    def rms_norm(
        self, value: np.ndarray, weight: np.ndarray | None = None, epsilon: float = 1e-5
    ) -> np.ndarray:
        value = _matrix(value)
        weight = (
            np.ones(value.shape[1], dtype=np.float32)
            if weight is None
            else _vector(weight, "weight")
        )
        if weight.shape[0] != value.shape[1]:
            raise ValueError("RMSNorm weight shape does not match input width")
        epsilon32 = np.float32(epsilon)
        if not np.isfinite(epsilon32) or epsilon32 <= 0:
            raise ValueError("epsilon must be finite and positive")
        values64 = value.astype(np.float64)
        inverse_rms = 1.0 / np.sqrt(np.mean(values64 * values64, axis=1, keepdims=True) + epsilon32)
        result = values64 * inverse_rms * weight.astype(np.float64)
        if not np.isfinite(result).all():
            raise FloatingPointError("RMSNorm produced NaN or infinite values")
        return np.asarray(result, dtype=np.float32)

    def apply_rope(
        self,
        query: np.ndarray,
        key: np.ndarray,
        positions: np.ndarray,
        rotary_dim: int | None = None,
        theta: float = 10000.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        query = _tensor3(query, "query")
        key = _tensor3(key, "key")
        if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
            raise ValueError("RoPE query and key shapes do not match")
        positions = _positions(positions, query.shape[0])
        rotary_dim = query.shape[2] if rotary_dim is None else int(rotary_dim)
        theta32 = np.float32(theta)
        if rotary_dim <= 0 or rotary_dim > query.shape[2] or rotary_dim % 2:
            raise ValueError("rotary_dim must be positive, even, and no larger than head size")
        if not np.isfinite(theta32) or theta32 <= 0:
            raise ValueError("RoPE theta must be finite and positive")
        query_output = query.copy()
        key_output = key.copy()
        for dimension in range(0, rotary_dim, 2):
            angles = np.array(
                [
                    _rope_angle(int(position), dimension, rotary_dim, float(theta32))
                    for position in positions
                ],
                dtype=np.float64,
            )
            cosine = np.cos(angles)[:, None]
            sine = np.sin(angles)[:, None]
            for tensor, output in ((query, query_output), (key, key_output)):
                first = tensor[:, :, dimension].astype(np.float64)
                second = tensor[:, :, dimension + 1].astype(np.float64)
                output[:, :, dimension] = first * cosine - second * sine
                output[:, :, dimension + 1] = first * sine + second * cosine
        if not np.isfinite(query_output).all() or not np.isfinite(key_output).all():
            raise FloatingPointError("RoPE produced NaN or infinite values")
        return query_output, key_output

    def causal_gqa(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        scale: float | None = None,
    ) -> np.ndarray:
        query, key, value = _attention_shapes(query, key, value)
        scale32 = _scale(scale, query.shape[2])
        output = np.empty_like(query)
        heads_per_group = query.shape[1] // key.shape[1]
        for token in range(query.shape[0]):
            for query_head in range(query.shape[1]):
                key_head = query_head // heads_per_group
                scores = (
                    key[: token + 1, key_head].astype(np.float64)
                    @ query[token, query_head].astype(np.float64)
                ) * float(scale32)
                scores -= scores.max()
                weights = np.exp(scores)
                weights /= weights.sum()
                output[token, query_head] = weights @ value[: token + 1, key_head].astype(np.float64)
        if not np.isfinite(output).all():
            raise FloatingPointError("causal GQA produced NaN or infinite values")
        return output

    def decode_attention(
        self,
        query: np.ndarray,
        key_cache: np.ndarray,
        value_cache: np.ndarray,
        position: int,
        scale: float | None = None,
    ) -> np.ndarray:
        query, key_cache, value_cache = _decode_shapes(query, key_cache, value_cache)
        if not isinstance(position, (int, np.integer)) or not 0 <= int(position) <= _UINT64_MASK:
            raise ValueError("position must fit uint64")
        scale32 = _scale(scale, query.shape[1])
        visible = min(key_cache.shape[0], int(position) + 1)
        output = np.empty_like(query)
        heads_per_group = query.shape[0] // key_cache.shape[1]
        for query_head in range(query.shape[0]):
            key_head = query_head // heads_per_group
            scores = (
                key_cache[:visible, key_head].astype(np.float64)
                @ query[query_head].astype(np.float64)
            ) * float(scale32)
            scores -= scores.max()
            weights = np.exp(scores)
            weights /= weights.sum()
            output[query_head] = weights @ value_cache[:visible, key_head].astype(np.float64)
        if not np.isfinite(output).all():
            raise FloatingPointError("decode attention produced NaN or infinite values")
        return output

    def quantize_q8(self, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value = _matrix(value)
        maximum = np.max(np.abs(value), axis=1)
        scales = np.where(
            maximum == 0,
            np.float32(1.0),
            np.maximum(maximum / np.float32(127.0), _FLOAT_MIN_NORMAL),
        ).astype(np.float32)
        scaled = value / scales[:, None]
        rounded = np.copysign(np.floor(np.abs(scaled) + np.float32(0.5)), scaled)
        quantized = np.clip(rounded, -127, 127).astype(np.int8)
        return quantized, scales

    def dequantize_q8(self, value: np.ndarray, scales: np.ndarray) -> np.ndarray:
        value = np.ascontiguousarray(value, dtype=np.int8)
        if value.ndim != 2 or any(size <= 0 for size in value.shape):
            raise ValueError("Q8 value must be a non-empty two-dimensional int8 tensor")
        scales = _vector(scales, "scales")
        if scales.shape[0] != value.shape[0]:
            raise ValueError("Q8 scale shape does not match input rows")
        if np.any(scales < _FLOAT_MIN_NORMAL):
            raise ValueError("Q8 scales must be positive normal float32 values")
        output = value.astype(np.float32) * scales[:, None]
        if not np.isfinite(output).all():
            raise FloatingPointError("Q8 dequantization produced NaN or infinite values")
        return output

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int = 0,
        counter: int = 0,
    ) -> int:
        logits, temperature32, top_k, top_p32, seed, counter = _sampling_arguments(
            logits, temperature, top_k, top_p, seed, counter
        )
        if temperature32 == 0:
            return int(np.argmax(logits))
        with np.errstate(over="ignore"):
            scaled = np.asarray(logits / temperature32, dtype=np.float32)
        if not np.isfinite(scaled).all():
            raise FloatingPointError("temperature scaling produced NaN or infinite logits")
        order = np.lexsort((np.arange(logits.size), -scaled.astype(np.float64)))
        if top_k:
            order = order[:top_k]
        weights = np.exp(scaled[order].astype(np.float64) - float(scaled[order[0]]))
        total = float(weights.sum())
        cumulative = np.cumsum(weights)
        kept = int(np.searchsorted(cumulative / total, float(top_p32), side="left")) + 1
        kept = min(kept, len(order))
        target = _uniform_sample(seed, counter) * float(cumulative[kept - 1])
        selected = int(np.searchsorted(cumulative[:kept], target, side="right"))
        return int(order[min(selected, kept - 1)])

    def neuron_group(
        self,
        inputs: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
        gate_weights: np.ndarray,
        gate_bias: np.ndarray,
        residual: np.ndarray,
        residual_weights: np.ndarray,
        group_size: int,
    ) -> np.ndarray:
        inputs = _matrix(inputs, "inputs")
        weights = _matrix(weights, "weights")
        bias = _vector(bias, "bias")
        gate_weights = _matrix(gate_weights, "gate_weights")
        gate_bias = _vector(gate_bias, "gate_bias")
        residual = _matrix(residual, "residual")
        residual_weights = _matrix(residual_weights, "residual_weights")
        if (
            weights.shape != gate_weights.shape
            or inputs.shape[1] != weights.shape[0]
            or bias.shape[0] != weights.shape[1]
            or gate_bias.shape[0] != weights.shape[1]
            or residual.shape[0] != inputs.shape[0]
            or residual.shape[1] != residual_weights.shape[0]
            or residual_weights.shape[1] != weights.shape[1]
        ):
            raise ValueError("neuron group tensor shapes do not match")
        with np.errstate(over="ignore", invalid="ignore"):
            projection = self.matmul(inputs, weights) + bias
            gate_pre = self.matmul(inputs, gate_weights) + gate_bias
        if not np.isfinite(projection).all() or not np.isfinite(gate_pre).all():
            raise FloatingPointError("neuron group projection produced non-finite values")
        gate = _sigmoid(gate_pre)
        shortcut = self.matmul(residual, residual_weights)
        output = _mixed_activation(projection, group_size) * gate + shortcut
        if not np.isfinite(output).all():
            raise FloatingPointError("neuron group produced NaN or infinite values")
        return output

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "native": self.native,
            "abi": ABI_VERSION,
            "precision": "float32",
            "kernels": [
                "matmul",
                "softmax",
                "layer_norm",
                "rms_norm",
                "rope",
                "causal_gqa",
                "decode_attention",
                "q8",
                "sampler",
                "neuron_group",
            ],
        }


class CppBackend(TensorBackend):
    name = "cpp"
    native = True

    def __init__(self) -> None:
        from . import _native

        info = dict(_native.backend_info())
        if info.get("abi") != ABI_VERSION:
            raise RuntimeError(
                f"Momo-LM C++ backend ABI {info.get('abi')} is incompatible with {ABI_VERSION}"
            )
        self.module = _native

    @staticmethod
    def _result(buffer: bytearray, shape: tuple[int, ...]) -> np.ndarray:
        return np.frombuffer(buffer, dtype=np.float32).reshape(shape)

    def matmul(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left = _matrix(left, "left")
        right = _matrix(right, "right")
        if left.shape[1] != right.shape[0]:
            raise ValueError("matmul inner dimensions do not match")
        return self._result(self.module.matmul(left, right), (left.shape[0], right.shape[1]))

    def softmax(self, value: np.ndarray) -> np.ndarray:
        value = _matrix(value)
        return self._result(self.module.softmax(value), value.shape)

    def layer_norm(self, value: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        value = _matrix(value)
        return self._result(self.module.layer_norm(value, epsilon), value.shape)

    def rms_norm(
        self, value: np.ndarray, weight: np.ndarray | None = None, epsilon: float = 1e-5
    ) -> np.ndarray:
        value = _matrix(value)
        weight = (
            np.ones(value.shape[1], dtype=np.float32)
            if weight is None
            else _vector(weight, "weight")
        )
        if weight.shape[0] != value.shape[1]:
            raise ValueError("RMSNorm weight shape does not match input width")
        return self._result(self.module.rms_norm(value, weight, epsilon), value.shape)

    def apply_rope(
        self,
        query: np.ndarray,
        key: np.ndarray,
        positions: np.ndarray,
        rotary_dim: int | None = None,
        theta: float = 10000.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        query = _tensor3(query, "query")
        key = _tensor3(key, "key")
        if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
            raise ValueError("RoPE query and key shapes do not match")
        positions = _positions(positions, query.shape[0])
        rotary_dim = query.shape[2] if rotary_dim is None else int(rotary_dim)
        query_buffer, key_buffer = self.module.rope(
            query, key, positions, rotary_dim, theta
        )
        return self._result(query_buffer, query.shape), self._result(key_buffer, key.shape)

    def causal_gqa(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        scale: float | None = None,
    ) -> np.ndarray:
        query, key, value = _attention_shapes(query, key, value)
        scale32 = _scale(scale, query.shape[2])
        return self._result(self.module.causal_gqa(query, key, value, float(scale32)), query.shape)

    def decode_attention(
        self,
        query: np.ndarray,
        key_cache: np.ndarray,
        value_cache: np.ndarray,
        position: int,
        scale: float | None = None,
    ) -> np.ndarray:
        query, key_cache, value_cache = _decode_shapes(query, key_cache, value_cache)
        if not isinstance(position, (int, np.integer)) or not 0 <= int(position) <= _UINT64_MASK:
            raise ValueError("position must fit uint64")
        scale32 = _scale(scale, query.shape[1])
        buffer = self.module.decode_attention(
            query, key_cache, value_cache, int(position), float(scale32)
        )
        return self._result(buffer, query.shape)

    def quantize_q8(self, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value = _matrix(value)
        quantized, scales = self.module.quantize_q8(value)
        return (
            np.frombuffer(quantized, dtype=np.int8).reshape(value.shape),
            np.frombuffer(scales, dtype=np.float32).reshape((value.shape[0],)),
        )

    def dequantize_q8(self, value: np.ndarray, scales: np.ndarray) -> np.ndarray:
        value = np.ascontiguousarray(value, dtype=np.int8)
        if value.ndim != 2 or any(size <= 0 for size in value.shape):
            raise ValueError("Q8 value must be a non-empty two-dimensional int8 tensor")
        scales = _vector(scales, "scales")
        if scales.shape[0] != value.shape[0]:
            raise ValueError("Q8 scale shape does not match input rows")
        return self._result(self.module.dequantize_q8(value, scales), value.shape)

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int = 0,
        counter: int = 0,
    ) -> int:
        arguments = _sampling_arguments(logits, temperature, top_k, top_p, seed, counter)
        return int(self.module.sample(*arguments))

    def neuron_group(
        self,
        inputs: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
        gate_weights: np.ndarray,
        gate_bias: np.ndarray,
        residual: np.ndarray,
        residual_weights: np.ndarray,
        group_size: int,
    ) -> np.ndarray:
        values = (
            _matrix(inputs, "inputs"),
            _matrix(weights, "weights"),
            _vector(bias, "bias"),
            _matrix(gate_weights, "gate_weights"),
            _vector(gate_bias, "gate_bias"),
            _matrix(residual, "residual"),
            _matrix(residual_weights, "residual_weights"),
        )
        buffer = self.module.neuron_group(*values, group_size)
        return self._result(buffer, (values[0].shape[0], values[1].shape[1]))

    def describe(self) -> dict[str, Any]:
        result = dict(self.module.backend_info())
        result.update({"name": self.name, "native": True, "engine": result.pop("name", "momo-cpp")})
        return result


class RustBackend(TensorBackend):
    name = "rust"
    native = True

    def __init__(self, path: Path) -> None:
        self.path = path
        self.library = ctypes.CDLL(str(path))
        pointer = ctypes.POINTER(ctypes.c_float)
        int8_pointer = ctypes.POINTER(ctypes.c_int8)
        uint64_pointer = ctypes.POINTER(ctypes.c_uint64)
        size_pointer = ctypes.POINTER(ctypes.c_size_t)
        size = ctypes.c_size_t
        self.library.momo_rust_abi_version.restype = ctypes.c_int
        self.library.momo_rust_backend_name.restype = ctypes.c_char_p
        self.library.momo_rust_matmul_f32.argtypes = [pointer, pointer, pointer, size, size, size]
        self.library.momo_rust_matmul_f32.restype = ctypes.c_int
        self.library.momo_rust_softmax_f32.argtypes = [pointer, pointer, size, size]
        self.library.momo_rust_softmax_f32.restype = ctypes.c_int
        self.library.momo_rust_layer_norm_f32.argtypes = [
            pointer,
            pointer,
            size,
            size,
            ctypes.c_float,
        ]
        self.library.momo_rust_layer_norm_f32.restype = ctypes.c_int
        self.library.momo_rust_rms_norm_f32.argtypes = [
            pointer,
            pointer,
            pointer,
            size,
            size,
            ctypes.c_float,
        ]
        self.library.momo_rust_rms_norm_f32.restype = ctypes.c_int
        self.library.momo_rust_rope_f32.argtypes = [
            pointer,
            pointer,
            uint64_pointer,
            pointer,
            pointer,
            size,
            size,
            size,
            size,
            size,
            ctypes.c_float,
        ]
        self.library.momo_rust_rope_f32.restype = ctypes.c_int
        self.library.momo_rust_causal_gqa_f32.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            size,
            size,
            size,
            size,
            ctypes.c_float,
        ]
        self.library.momo_rust_causal_gqa_f32.restype = ctypes.c_int
        self.library.momo_rust_decode_attention_f32.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            size,
            size,
            size,
            size,
            ctypes.c_uint64,
            ctypes.c_float,
        ]
        self.library.momo_rust_decode_attention_f32.restype = ctypes.c_int
        self.library.momo_rust_quantize_q8_f32.argtypes = [
            pointer,
            int8_pointer,
            pointer,
            size,
            size,
        ]
        self.library.momo_rust_quantize_q8_f32.restype = ctypes.c_int
        self.library.momo_rust_dequantize_q8_f32.argtypes = [
            int8_pointer,
            pointer,
            pointer,
            size,
            size,
        ]
        self.library.momo_rust_dequantize_q8_f32.restype = ctypes.c_int
        self.library.momo_rust_sample_f32.argtypes = [
            pointer,
            size,
            ctypes.c_float,
            size,
            ctypes.c_float,
            ctypes.c_uint64,
            ctypes.c_uint64,
            size_pointer,
        ]
        self.library.momo_rust_sample_f32.restype = ctypes.c_int
        self.library.momo_rust_neuron_group_f32.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            size,
            size,
            size,
            size,
            size,
        ]
        self.library.momo_rust_neuron_group_f32.restype = ctypes.c_int
        if self.library.momo_rust_abi_version() != ABI_VERSION:
            raise RuntimeError("Momo-LM Rust backend ABI is incompatible")

    @staticmethod
    def _pointer(value: np.ndarray) -> Any:
        return value.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    @staticmethod
    def _int8_pointer(value: np.ndarray) -> Any:
        return value.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))

    @staticmethod
    def _uint64_pointer(value: np.ndarray) -> Any:
        return value.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status == -1:
            raise ValueError(f"Rust backend {operation} rejected invalid arguments")
        if status == -4:
            raise MemoryError(f"Rust backend {operation} ran out of memory")
        if status == -3:
            raise OverflowError(f"Rust backend {operation} rejected an overflowing shape")
        if status == -2:
            raise FloatingPointError(f"Rust backend {operation} rejected non-finite arithmetic")
        if status != 0:
            raise RuntimeError(f"Rust backend {operation} failed with status {status}")

    def matmul(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left = _matrix(left, "left")
        right = _matrix(right, "right")
        if left.shape[1] != right.shape[0]:
            raise ValueError("matmul inner dimensions do not match")
        output = np.empty((left.shape[0], right.shape[1]), dtype=np.float32)
        status = self.library.momo_rust_matmul_f32(
            self._pointer(left),
            self._pointer(right),
            self._pointer(output),
            left.shape[0],
            left.shape[1],
            right.shape[1],
        )
        self._check(status, "matmul")
        return output

    def softmax(self, value: np.ndarray) -> np.ndarray:
        value = _matrix(value)
        output = np.empty_like(value)
        status = self.library.momo_rust_softmax_f32(
            self._pointer(value), self._pointer(output), value.shape[0], value.shape[1]
        )
        self._check(status, "softmax")
        return output

    def layer_norm(self, value: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        value = _matrix(value)
        output = np.empty_like(value)
        status = self.library.momo_rust_layer_norm_f32(
            self._pointer(value),
            self._pointer(output),
            value.shape[0],
            value.shape[1],
            epsilon,
        )
        self._check(status, "layer_norm")
        return output

    def rms_norm(
        self, value: np.ndarray, weight: np.ndarray | None = None, epsilon: float = 1e-5
    ) -> np.ndarray:
        value = _matrix(value)
        weight = (
            np.ones(value.shape[1], dtype=np.float32)
            if weight is None
            else _vector(weight, "weight")
        )
        if weight.shape[0] != value.shape[1]:
            raise ValueError("RMSNorm weight shape does not match input width")
        output = np.empty_like(value)
        status = self.library.momo_rust_rms_norm_f32(
            self._pointer(value),
            self._pointer(weight),
            self._pointer(output),
            value.shape[0],
            value.shape[1],
            epsilon,
        )
        self._check(status, "rms_norm")
        return output

    def apply_rope(
        self,
        query: np.ndarray,
        key: np.ndarray,
        positions: np.ndarray,
        rotary_dim: int | None = None,
        theta: float = 10000.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        query = _tensor3(query, "query")
        key = _tensor3(key, "key")
        if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
            raise ValueError("RoPE query and key shapes do not match")
        positions = _positions(positions, query.shape[0])
        rotary_dim = query.shape[2] if rotary_dim is None else int(rotary_dim)
        query_output = np.empty_like(query)
        key_output = np.empty_like(key)
        status = self.library.momo_rust_rope_f32(
            self._pointer(query),
            self._pointer(key),
            self._uint64_pointer(positions),
            self._pointer(query_output),
            self._pointer(key_output),
            query.shape[0],
            query.shape[1],
            key.shape[1],
            query.shape[2],
            rotary_dim,
            theta,
        )
        self._check(status, "rope")
        return query_output, key_output

    def causal_gqa(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        scale: float | None = None,
    ) -> np.ndarray:
        query, key, value = _attention_shapes(query, key, value)
        scale32 = _scale(scale, query.shape[2])
        output = np.empty_like(query)
        status = self.library.momo_rust_causal_gqa_f32(
            self._pointer(query),
            self._pointer(key),
            self._pointer(value),
            self._pointer(output),
            query.shape[0],
            query.shape[1],
            key.shape[1],
            query.shape[2],
            scale32,
        )
        self._check(status, "causal_gqa")
        return output

    def decode_attention(
        self,
        query: np.ndarray,
        key_cache: np.ndarray,
        value_cache: np.ndarray,
        position: int,
        scale: float | None = None,
    ) -> np.ndarray:
        query, key_cache, value_cache = _decode_shapes(query, key_cache, value_cache)
        if not isinstance(position, (int, np.integer)) or not 0 <= int(position) <= _UINT64_MASK:
            raise ValueError("position must fit uint64")
        scale32 = _scale(scale, query.shape[1])
        output = np.empty_like(query)
        status = self.library.momo_rust_decode_attention_f32(
            self._pointer(query),
            self._pointer(key_cache),
            self._pointer(value_cache),
            self._pointer(output),
            key_cache.shape[0],
            query.shape[0],
            key_cache.shape[1],
            query.shape[1],
            int(position),
            scale32,
        )
        self._check(status, "decode_attention")
        return output

    def quantize_q8(self, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value = _matrix(value)
        output = np.empty(value.shape, dtype=np.int8)
        scales = np.empty(value.shape[0], dtype=np.float32)
        status = self.library.momo_rust_quantize_q8_f32(
            self._pointer(value),
            self._int8_pointer(output),
            self._pointer(scales),
            value.shape[0],
            value.shape[1],
        )
        self._check(status, "quantize_q8")
        return output, scales

    def dequantize_q8(self, value: np.ndarray, scales: np.ndarray) -> np.ndarray:
        value = np.ascontiguousarray(value, dtype=np.int8)
        if value.ndim != 2 or any(size <= 0 for size in value.shape):
            raise ValueError("Q8 value must be a non-empty two-dimensional int8 tensor")
        scales = _vector(scales, "scales")
        if scales.shape[0] != value.shape[0]:
            raise ValueError("Q8 scale shape does not match input rows")
        output = np.empty(value.shape, dtype=np.float32)
        status = self.library.momo_rust_dequantize_q8_f32(
            self._int8_pointer(value),
            self._pointer(scales),
            self._pointer(output),
            value.shape[0],
            value.shape[1],
        )
        self._check(status, "dequantize_q8")
        return output

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int = 0,
        counter: int = 0,
    ) -> int:
        logits, temperature32, top_k, top_p32, seed, counter = _sampling_arguments(
            logits, temperature, top_k, top_p, seed, counter
        )
        sampled = ctypes.c_size_t()
        status = self.library.momo_rust_sample_f32(
            self._pointer(logits),
            logits.size,
            temperature32,
            top_k,
            top_p32,
            seed,
            counter,
            ctypes.byref(sampled),
        )
        self._check(status, "sample")
        return int(sampled.value)

    def neuron_group(
        self,
        inputs: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
        gate_weights: np.ndarray,
        gate_bias: np.ndarray,
        residual: np.ndarray,
        residual_weights: np.ndarray,
        group_size: int,
    ) -> np.ndarray:
        inputs = _matrix(inputs, "inputs")
        weights = _matrix(weights, "weights")
        bias = _vector(bias, "bias")
        gate_weights = _matrix(gate_weights, "gate_weights")
        gate_bias = _vector(gate_bias, "gate_bias")
        residual = _matrix(residual, "residual")
        residual_weights = _matrix(residual_weights, "residual_weights")
        if (
            weights.shape != gate_weights.shape
            or inputs.shape[1] != weights.shape[0]
            or bias.shape[0] != weights.shape[1]
            or gate_bias.shape[0] != weights.shape[1]
            or residual.shape[0] != inputs.shape[0]
            or residual.shape[1] != residual_weights.shape[0]
            or residual_weights.shape[1] != weights.shape[1]
            or not isinstance(group_size, (int, np.integer))
            or int(group_size) <= 0
        ):
            raise ValueError("neuron group tensor shapes do not match")
        output = np.empty((inputs.shape[0], weights.shape[1]), dtype=np.float32)
        status = self.library.momo_rust_neuron_group_f32(
            self._pointer(inputs),
            self._pointer(weights),
            self._pointer(bias),
            self._pointer(gate_weights),
            self._pointer(gate_bias),
            self._pointer(residual),
            self._pointer(residual_weights),
            self._pointer(output),
            inputs.shape[0],
            inputs.shape[1],
            weights.shape[1],
            residual.shape[1],
            group_size,
        )
        self._check(status, "neuron_group")
        return output

    def describe(self) -> dict[str, Any]:
        engine = self.library.momo_rust_backend_name().decode("utf-8")
        result = super().describe()
        result.update(
            {
                "name": self.name,
                "native": True,
                "engine": engine,
                "library": str(self.path),
            }
        )
        return result


class HybridBackend(TensorBackend):
    name = "hybrid"

    def __init__(self, backends: list[TensorBackend]) -> None:
        if not backends:
            raise ValueError("hybrid backend requires at least one route")
        self.backends = tuple(backends)
        self.native = any(backend.native for backend in backends)
        self.name = backends[0].name

    def _route(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        errors: list[str] = []
        for backend in self.backends:
            method: Callable[..., Any] = getattr(backend, operation)
            try:
                return method(*args, **kwargs)
            except (ImportError, OSError, RuntimeError) as exc:
                errors.append(f"{backend.name}: {exc}")
        raise RuntimeError(f"all {operation} routes failed: {'; '.join(errors)}")

    def matmul(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._route("matmul", *args, **kwargs)

    def softmax(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._route("softmax", *args, **kwargs)

    def layer_norm(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._route("layer_norm", *args, **kwargs)

    def rms_norm(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._route("rms_norm", *args, **kwargs)

    def apply_rope(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        return self._route("apply_rope", *args, **kwargs)

    def causal_gqa(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._route("causal_gqa", *args, **kwargs)

    def decode_attention(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._route("decode_attention", *args, **kwargs)

    def quantize_q8(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        return self._route("quantize_q8", *args, **kwargs)

    def dequantize_q8(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._route("dequantize_q8", *args, **kwargs)

    def sample(self, *args: Any, **kwargs: Any) -> int:
        return self._route("sample", *args, **kwargs)

    def neuron_group(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._route("neuron_group", *args, **kwargs)

    def describe(self) -> dict[str, Any]:
        result = super().describe()
        result.update(
            {
                "name": self.name,
                "native": self.native,
                "routing": "hybrid",
                "routes": [backend.name for backend in self.backends],
                "engines": [backend.describe() for backend in self.backends],
            }
        )
        return result


def _rust_candidates() -> list[Path]:
    configured = os.environ.get("MOMO_RUST_LIBRARY")
    names = {
        "win32": "momo_core_rust.dll",
        "cygwin": "momo_core_rust.dll",
        "darwin": "libmomo_core_rust.dylib",
    }
    packaged = Path(__file__).resolve().parent / "_native_libs" / names.get(
        sys.platform, "libmomo_core_rust.so"
    )
    return ([Path(configured).expanduser()] if configured else []) + [packaged]


def _native_routes(errors: list[str]) -> list[TensorBackend]:
    routes: list[TensorBackend] = []
    for candidate in _rust_candidates():
        if candidate.is_file():
            try:
                routes.append(RustBackend(candidate))
                break
            except (AttributeError, OSError, RuntimeError) as exc:
                errors.append(f"rust: {exc}")
    try:
        routes.append(CppBackend())
    except (AttributeError, ImportError, OSError, RuntimeError) as exc:
        errors.append(f"cpp: {exc}")
    return routes


@lru_cache(maxsize=1)
def get_backend() -> TensorBackend:
    preference = os.environ.get("MOMO_BACKEND", "auto").strip().lower()
    require_native = os.environ.get("MOMO_REQUIRE_NATIVE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    valid = {"auto", "hybrid", "rust", "cpp", "native", "numpy", "python"}
    if preference not in valid:
        raise ValueError(f"unknown MOMO_BACKEND value: {preference}")
    if require_native and preference in {"numpy", "python"}:
        raise RuntimeError("MOMO_REQUIRE_NATIVE cannot be combined with a NumPy-only backend")
    errors: list[str] = []
    if preference == "rust":
        for candidate in _rust_candidates():
            if candidate.is_file():
                try:
                    return RustBackend(candidate)
                except (AttributeError, OSError, RuntimeError) as exc:
                    errors.append(str(exc))
        raise RuntimeError(
            "requested Rust backend is unavailable"
            + (f": {'; '.join(errors)}" if errors else "")
        )
    if preference == "cpp":
        try:
            return CppBackend()
        except (AttributeError, ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(f"requested C++ backend is unavailable: {exc}") from None
    if preference in {"auto", "hybrid", "native"}:
        routes = _native_routes(errors)
        if routes:
            return HybridBackend(routes + [TensorBackend()])
        if preference in {"hybrid", "native"} or require_native:
            detail = f": {'; '.join(errors)}" if errors else ""
            raise RuntimeError(f"no native Momo-LM ABI {ABI_VERSION} backend is available{detail}")
    return TensorBackend()


def reset_backend() -> None:
    get_backend.cache_clear()
