import os
import unittest

import numpy as np

from momo_lm.backend import TensorBackend, get_backend


class BackendTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(9)
        self.left = rng.normal(size=(5, 7)).astype(np.float32)
        self.right = rng.normal(size=(7, 4)).astype(np.float32)
        self.reference = TensorBackend()
        self.backend = get_backend()

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

    def test_required_native_environment_uses_native_backend(self) -> None:
        if os.environ.get("MOMO_REQUIRE_NATIVE") == "1":
            self.assertTrue(self.backend.native)


if __name__ == "__main__":
    unittest.main()
