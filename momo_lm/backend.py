from __future__ import annotations

import ctypes
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


def _matrix(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError("native tensor operations require two-dimensional matrices")
    return result


def _vector(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    if result.ndim != 1:
        raise ValueError("native neuron biases require one-dimensional vectors")
    return result


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _mixed_activation(value: np.ndarray, group_size: int) -> np.ndarray:
    output = np.empty_like(value)
    for start in range(0, value.shape[1], group_size):
        stop = min(start + group_size, value.shape[1])
        group = start // group_size
        chunk = value[:, start:stop]
        if group % 3 == 0:
            output[:, start:stop] = np.tanh(chunk)
        elif group % 3 == 1:
            transformed = np.sqrt(2.0 / np.pi) * (chunk + 0.044715 * chunk**3)
            output[:, start:stop] = 0.5 * chunk * (1.0 + np.tanh(transformed))
        else:
            output[:, start:stop] = chunk * _sigmoid(chunk)
    return output


class TensorBackend:
    name = "numpy"
    native = False

    def matmul(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return _matrix(left) @ _matrix(right)

    def softmax(self, value: np.ndarray) -> np.ndarray:
        value = _matrix(value)
        shifted = value - value.max(axis=1, keepdims=True)
        exponential = np.exp(shifted)
        return exponential / exponential.sum(axis=1, keepdims=True)

    def layer_norm(self, value: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        value = _matrix(value)
        mean = value.mean(axis=1, keepdims=True)
        variance = value.var(axis=1, keepdims=True)
        return (value - mean) / np.sqrt(variance + epsilon)

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
        projection = self.matmul(inputs, weights) + _vector(bias)
        gate = _sigmoid(self.matmul(inputs, gate_weights) + _vector(gate_bias))
        shortcut = self.matmul(residual, residual_weights)
        return _mixed_activation(projection, group_size) * gate + shortcut

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "native": self.native, "abi": 1, "precision": "float32"}


class CppBackend(TensorBackend):
    name = "cpp"
    native = True

    def __init__(self) -> None:
        from . import _native

        self.module = _native

    @staticmethod
    def _result(buffer: bytearray, shape: tuple[int, int]) -> np.ndarray:
        return np.frombuffer(buffer, dtype=np.float32).reshape(shape)

    def matmul(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left = _matrix(left)
        right = _matrix(right)
        if left.shape[1] != right.shape[0]:
            raise ValueError("matmul inner dimensions do not match")
        return self._result(self.module.matmul(left, right), (left.shape[0], right.shape[1]))

    def softmax(self, value: np.ndarray) -> np.ndarray:
        value = _matrix(value)
        return self._result(self.module.softmax(value), value.shape)

    def layer_norm(self, value: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        value = _matrix(value)
        return self._result(self.module.layer_norm(value, epsilon), value.shape)

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
            _matrix(inputs),
            _matrix(weights),
            _vector(bias),
            _matrix(gate_weights),
            _vector(gate_bias),
            _matrix(residual),
            _matrix(residual_weights),
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
        size = ctypes.c_size_t
        self.library.momo_rust_abi_version.restype = ctypes.c_int
        self.library.momo_rust_backend_name.restype = ctypes.c_char_p
        self.library.momo_rust_matmul_f32.argtypes = [pointer, pointer, pointer, size, size, size]
        self.library.momo_rust_matmul_f32.restype = ctypes.c_int
        self.library.momo_rust_softmax_f32.argtypes = [pointer, pointer, size, size]
        self.library.momo_rust_softmax_f32.restype = ctypes.c_int
        self.library.momo_rust_layer_norm_f32.argtypes = [pointer, pointer, size, size, ctypes.c_float]
        self.library.momo_rust_layer_norm_f32.restype = ctypes.c_int
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
        if self.library.momo_rust_abi_version() != 1:
            raise RuntimeError("Momo-LM Rust backend ABI is incompatible")

    @staticmethod
    def _pointer(value: np.ndarray) -> Any:
        return value.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"Rust backend {operation} failed with status {status}")

    def matmul(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left = _matrix(left)
        right = _matrix(right)
        if left.shape[1] != right.shape[0]:
            raise ValueError("matmul inner dimensions do not match")
        output = np.empty((left.shape[0], right.shape[1]), dtype=np.float32)
        status = self.library.momo_rust_matmul_f32(
            self._pointer(left), self._pointer(right), self._pointer(output),
            left.shape[0], left.shape[1], right.shape[1],
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
            self._pointer(value), self._pointer(output), value.shape[0], value.shape[1], epsilon
        )
        self._check(status, "layer_norm")
        return output

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
        inputs = _matrix(inputs)
        weights = _matrix(weights)
        bias = _vector(bias)
        gate_weights = _matrix(gate_weights)
        gate_bias = _vector(gate_bias)
        residual = _matrix(residual)
        residual_weights = _matrix(residual_weights)
        output = np.empty((inputs.shape[0], weights.shape[1]), dtype=np.float32)
        status = self.library.momo_rust_neuron_group_f32(
            self._pointer(inputs), self._pointer(weights), self._pointer(bias),
            self._pointer(gate_weights), self._pointer(gate_bias), self._pointer(residual),
            self._pointer(residual_weights), self._pointer(output), inputs.shape[0],
            inputs.shape[1], weights.shape[1], residual.shape[1], group_size,
        )
        self._check(status, "neuron_group")
        return output

    def describe(self) -> dict[str, Any]:
        engine = self.library.momo_rust_backend_name().decode("utf-8")
        return {
            "name": self.name,
            "native": True,
            "engine": engine,
            "abi": self.library.momo_rust_abi_version(),
            "precision": "float32",
            "library": str(self.path),
        }


def _rust_candidates() -> list[Path]:
    configured = os.environ.get("MOMO_RUST_LIBRARY")
    names = {
        "win32": "momo_core_rust.dll",
        "cygwin": "momo_core_rust.dll",
        "darwin": "libmomo_core_rust.dylib",
    }
    packaged = Path(__file__).resolve().parent / "_native_libs" / names.get(sys.platform, "libmomo_core_rust.so")
    return ([Path(configured).expanduser()] if configured else []) + [packaged]


@lru_cache(maxsize=1)
def get_backend() -> TensorBackend:
    preference = os.environ.get("MOMO_BACKEND", "auto").strip().lower()
    require_native = os.environ.get("MOMO_REQUIRE_NATIVE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    errors: list[str] = []
    if preference in {"auto", "rust", "native"}:
        for candidate in _rust_candidates():
            if candidate.is_file():
                try:
                    return RustBackend(candidate)
                except (OSError, RuntimeError) as exc:
                    errors.append(str(exc))
        if preference == "rust":
            raise RuntimeError("requested Rust backend is unavailable" + (f": {'; '.join(errors)}" if errors else ""))
    if preference in {"auto", "cpp", "native"}:
        try:
            return CppBackend()
        except (ImportError, OSError) as exc:
            errors.append(str(exc))
        if preference == "cpp":
            raise RuntimeError(f"requested C++ backend is unavailable: {'; '.join(errors)}") from None
    if preference not in {"auto", "numpy", "python", "native"}:
        raise ValueError(f"unknown MOMO_BACKEND value: {preference}")
    if preference == "native" or (require_native and preference == "auto"):
        raise RuntimeError("no native Momo-LM backend is available" + (f": {'; '.join(errors)}" if errors else ""))
    return TensorBackend()


def reset_backend() -> None:
    get_backend.cache_clear()
