import os
import unittest
from unittest.mock import patch

import numpy as np

from momo_lm.backend import (
    ABI_VERSION,
    CppBackend,
    HybridBackend,
    RustBackend,
    TensorBackend,
    _rust_candidates,
    get_backend,
    reset_backend,
)


class BackendTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(9)
        self.left = rng.normal(size=(5, 7)).astype(np.float32)
        self.right = rng.normal(size=(7, 4)).astype(np.float32)
        self.reference = TensorBackend()
        self.backend = get_backend()

    def assert_extended_backend_matches(self, backend: TensorBackend) -> None:
        rng = np.random.default_rng(101)
        left = rng.normal(size=(3, 5)).astype(np.float32)
        right = rng.normal(size=(5, 4)).astype(np.float32)
        values = self.reference.matmul(left, right)
        np.testing.assert_allclose(backend.matmul(left, right), values, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(
            backend.softmax(values), self.reference.softmax(values), rtol=3e-5, atol=3e-5
        )
        np.testing.assert_allclose(
            backend.layer_norm(values),
            self.reference.layer_norm(values),
            rtol=3e-4,
            atol=3e-5,
        )
        np.testing.assert_allclose(
            backend.rms_norm(values),
            self.reference.rms_norm(values),
            rtol=3e-5,
            atol=3e-5,
        )
        query = rng.normal(size=(3, 2, 4)).astype(np.float32)
        key = rng.normal(size=(3, 1, 4)).astype(np.float32)
        value = rng.normal(size=(3, 1, 4)).astype(np.float32)
        positions = np.array([0, 1_000_000, 2**63 + 9], dtype=np.uint64)
        actual_query, actual_key = backend.apply_rope(query, key, positions)
        expected_query, expected_key = self.reference.apply_rope(query, key, positions)
        np.testing.assert_allclose(actual_query, expected_query, rtol=4e-5, atol=4e-5)
        np.testing.assert_allclose(actual_key, expected_key, rtol=4e-5, atol=4e-5)
        np.testing.assert_allclose(
            backend.causal_gqa(query, key, value),
            self.reference.causal_gqa(query, key, value),
            rtol=4e-5,
            atol=4e-5,
        )
        np.testing.assert_allclose(
            backend.decode_attention(query[-1], key, value, 2**64 - 1),
            self.reference.decode_attention(query[-1], key, value, 2**64 - 1),
            rtol=4e-5,
            atol=4e-5,
        )
        quantized, scales = backend.quantize_q8(values)
        expected_quantized, expected_scales = self.reference.quantize_q8(values)
        np.testing.assert_array_equal(quantized, expected_quantized)
        np.testing.assert_array_equal(scales, expected_scales)
        np.testing.assert_allclose(
            backend.dequantize_q8(quantized, scales),
            self.reference.dequantize_q8(expected_quantized, expected_scales),
        )
        logits = rng.normal(size=17).astype(np.float32)
        self.assertEqual(
            backend.sample(logits, 0.8, 7, 0.9, 31, 4),
            self.reference.sample(logits, 0.8, 7, 0.9, 31, 4),
        )
        inputs = rng.normal(size=(2, 5)).astype(np.float32)
        weights = rng.normal(size=(5, 6)).astype(np.float32)
        bias = rng.normal(size=6).astype(np.float32)
        gate_weights = rng.normal(size=(5, 6)).astype(np.float32)
        gate_bias = rng.normal(size=6).astype(np.float32)
        residual = rng.normal(size=(2, 3)).astype(np.float32)
        residual_weights = rng.normal(size=(3, 6)).astype(np.float32)
        arguments = (
            inputs,
            weights,
            bias,
            gate_weights,
            gate_bias,
            residual,
            residual_weights,
            2,
        )
        np.testing.assert_allclose(
            backend.neuron_group(*arguments),
            self.reference.neuron_group(*arguments),
            rtol=3e-4,
            atol=3e-5,
        )

    def test_matrix_softmax_and_layer_norm_match_reference(self) -> None:
        np.testing.assert_allclose(
            self.backend.matmul(self.left, self.right),
            self.reference.matmul(self.left, self.right),
            rtol=2e-5,
            atol=2e-5,
        )
        values = self.reference.matmul(self.left, self.right)
        np.testing.assert_allclose(
            self.backend.softmax(values), self.reference.softmax(values), rtol=2e-5, atol=2e-6
        )
        np.testing.assert_allclose(
            self.backend.layer_norm(values),
            self.reference.layer_norm(values),
            rtol=2e-4,
            atol=2e-5,
        )

    def test_rms_norm_and_high_position_rope_match_reference(self) -> None:
        rng = np.random.default_rng(17)
        values = rng.normal(size=(4, 8)).astype(np.float32)
        weight = rng.normal(size=8).astype(np.float32)
        np.testing.assert_allclose(
            self.backend.rms_norm(values, weight),
            self.reference.rms_norm(values, weight),
            rtol=3e-5,
            atol=3e-5,
        )
        query = rng.normal(size=(5, 4, 8)).astype(np.float32)
        key = rng.normal(size=(5, 2, 8)).astype(np.float32)
        positions = np.array([0, 1, 1_000_000, 2**53 + 3, 2**63 + 5], dtype=np.uint64)
        actual_query, actual_key = self.backend.apply_rope(query, key, positions)
        expected_query, expected_key = self.reference.apply_rope(query, key, positions)
        np.testing.assert_allclose(actual_query, expected_query, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(actual_key, expected_key, rtol=3e-5, atol=3e-5)
        repeated_query = np.repeat(query[:1], 2, axis=0)
        repeated_key = np.repeat(key[:1], 2, axis=0)
        adjacent, _ = self.backend.apply_rope(
            repeated_query,
            repeated_key,
            np.array([2**63 + 5, 2**63 + 6], dtype=np.uint64),
        )
        self.assertGreater(float(np.max(np.abs(adjacent[0] - adjacent[1]))), 1e-4)

    def test_causal_gqa_and_decode_match_reference(self) -> None:
        rng = np.random.default_rng(23)
        query = rng.normal(size=(6, 4, 8)).astype(np.float32)
        key = rng.normal(size=(6, 2, 8)).astype(np.float32)
        value = rng.normal(size=(6, 2, 8)).astype(np.float32)
        np.testing.assert_allclose(
            self.backend.causal_gqa(query, key, value),
            self.reference.causal_gqa(query, key, value),
            rtol=4e-5,
            atol=4e-5,
        )
        for position in (0, 3, 2**64 - 1):
            np.testing.assert_allclose(
                self.backend.decode_attention(query[-1], key, value, position),
                self.reference.decode_attention(query[-1], key, value, position),
                rtol=4e-5,
                atol=4e-5,
            )

    def test_q8_and_deterministic_sampler_match_reference(self) -> None:
        smallest = np.nextafter(np.float32(0), np.float32(1))
        values = np.array(
            [
                [0.0, 1.0, -2.0, 3.0],
                [smallest, -smallest, 0.0, 0.0],
                [127.0, 0.5, -0.5, 0.0],
            ],
            dtype=np.float32,
        )
        quantized, scales = self.backend.quantize_q8(values)
        expected_quantized, expected_scales = self.reference.quantize_q8(values)
        np.testing.assert_array_equal(quantized, expected_quantized)
        np.testing.assert_array_equal(scales, expected_scales)
        np.testing.assert_array_equal(quantized[2], np.array([127, 1, -1, 0], dtype=np.int8))
        self.assertTrue(np.all(scales >= np.finfo(np.float32).tiny))
        np.testing.assert_allclose(
            self.backend.dequantize_q8(quantized, scales),
            self.reference.dequantize_q8(expected_quantized, expected_scales),
        )
        logits = np.random.default_rng(29).normal(size=41).astype(np.float32)
        for seed in range(4):
            for counter in range(5):
                expected = self.reference.sample(logits, 0.73, 13, 0.91, seed, counter)
                self.assertEqual(
                    self.backend.sample(logits, 0.73, 13, 0.91, seed, counter), expected
                )
        self.assertEqual(self.backend.sample(logits, 0.0), int(np.argmax(logits)))

    def test_fused_neuron_groups_match_reference(self) -> None:
        rng = np.random.default_rng(11)
        inputs = rng.normal(size=(3, 8)).astype(np.float32)
        weights = rng.normal(size=(8, 12)).astype(np.float32)
        bias = rng.normal(size=12).astype(np.float32)
        gate_weights = rng.normal(size=(8, 12)).astype(np.float32)
        gate_bias = rng.normal(size=12).astype(np.float32)
        residual = rng.normal(size=(3, 5)).astype(np.float32)
        residual_weights = rng.normal(size=(5, 12)).astype(np.float32)
        arguments = (
            inputs,
            weights,
            bias,
            gate_weights,
            gate_bias,
            residual,
            residual_weights,
            4,
        )
        np.testing.assert_allclose(
            self.backend.neuron_group(*arguments),
            self.reference.neuron_group(*arguments),
            rtol=3e-4,
            atol=3e-5,
        )

    def test_invalid_shapes_and_non_finite_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.backend.matmul(np.ones((2, 3), np.float32), np.ones((4, 2), np.float32))
        with self.assertRaises(ValueError):
            self.backend.softmax(np.array([[0.0, np.inf]], dtype=np.float32))
        with self.assertRaises(ValueError):
            self.backend.causal_gqa(
                np.ones((2, 3, 4), np.float32),
                np.ones((2, 2, 4), np.float32),
                np.ones((2, 2, 4), np.float32),
            )
        with self.assertRaises(ValueError):
            self.backend.apply_rope(
                np.ones((2, 1, 4), np.float32),
                np.ones((2, 1, 4), np.float32),
                np.array([-1, 0]),
            )
        with self.assertRaises(ValueError):
            self.backend.sample(np.ones(4, np.float32), top_k=5)

    def test_backend_metadata_and_hybrid_fallback(self) -> None:
        description = self.backend.describe()
        self.assertEqual(description["abi"], ABI_VERSION)
        self.assertIn("causal_gqa", description["kernels"])

        class BrokenBackend(TensorBackend):
            name = "broken"

            def matmul(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
                raise RuntimeError("deliberate route failure")

        hybrid = HybridBackend([BrokenBackend(), self.reference])
        np.testing.assert_allclose(
            hybrid.matmul(self.left, self.right), self.reference.matmul(self.left, self.right)
        )

    def test_required_native_contract_cannot_select_numpy(self) -> None:
        with patch.dict(os.environ, {"MOMO_REQUIRE_NATIVE": "1", "MOMO_BACKEND": "numpy"}):
            reset_backend()
            with self.assertRaises(RuntimeError):
                get_backend()
        reset_backend()

    def test_required_native_environment_uses_native_backend(self) -> None:
        if os.environ.get("MOMO_REQUIRE_NATIVE") == "1":
            self.assertTrue(self.backend.native)

    def test_cpp_backend_is_real_when_native_is_required(self) -> None:
        try:
            backend = CppBackend()
        except (ImportError, OSError, RuntimeError) as exc:
            if os.environ.get("MOMO_REQUIRE_NATIVE") == "1":
                self.fail(f"required C++ backend is unavailable: {exc}")
            self.skipTest(f"optional C++ backend is unavailable: {exc}")
        self.assertEqual(backend.describe()["abi"], ABI_VERSION)
        self.assert_extended_backend_matches(backend)

    def test_rust_backend_is_real_when_native_is_required(self) -> None:
        error = "packaged Rust library is missing"
        for candidate in _rust_candidates():
            if candidate.is_file():
                try:
                    backend = RustBackend(candidate)
                except (AttributeError, OSError, RuntimeError) as exc:
                    error = str(exc)
                    continue
                self.assertEqual(backend.describe()["abi"], ABI_VERSION)
                self.assert_extended_backend_matches(backend)
                return
        if os.environ.get("MOMO_REQUIRE_NATIVE") == "1":
            self.fail(f"required Rust backend is unavailable: {error}")
        self.skipTest(f"optional Rust backend is unavailable: {error}")


if __name__ == "__main__":
    unittest.main()
