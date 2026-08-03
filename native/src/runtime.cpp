#include "momo_core.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace momo {

inline float sigmoid(float value) {
    if (value >= 0.0f) {
        const float exponential = std::exp(-value);
        return 1.0f / (1.0f + exponential);
    }
    const float exponential = std::exp(value);
    return exponential / (1.0f + exponential);
}

inline float activate(float value, std::size_t group) {
    switch (group % 3) {
        case 0:
            return std::tanh(value);
        case 1: {
            constexpr float coefficient = 0.7978845608028654f;
            const float cubic = value * value * value;
            return 0.5f * value * (1.0f + std::tanh(coefficient * (value + 0.044715f * cubic)));
        }
        default:
            return value * sigmoid(value);
    }
}

class NeuronGroupExecutor {
  public:
    int run(const float *input, const float *weights, const float *bias,
            const float *gate_weights, const float *gate_bias,
            const float *residual, const float *residual_weights, float *output,
            std::size_t batch, std::size_t input_size, std::size_t output_size,
            std::size_t residual_size, std::size_t group_size) const {
        if (input == nullptr || weights == nullptr || bias == nullptr ||
            gate_weights == nullptr || gate_bias == nullptr || residual == nullptr ||
            residual_weights == nullptr || output == nullptr || batch == 0 ||
            input_size == 0 || output_size == 0 || residual_size == 0 || group_size == 0) {
            return -1;
        }
        std::vector<float> projection(batch * output_size);
        std::vector<float> gate(batch * output_size);
        std::vector<float> shortcut(batch * output_size);
        if (momo_matmul_f32(input, weights, projection.data(), batch, input_size, output_size) != 0 ||
            momo_matmul_f32(input, gate_weights, gate.data(), batch, input_size, output_size) != 0 ||
            momo_matmul_f32(residual, residual_weights, shortcut.data(), batch, residual_size, output_size) != 0) {
            return -2;
        }
        for (std::size_t row = 0; row < batch; ++row) {
            for (std::size_t column = 0; column < output_size; ++column) {
                const std::size_t index = row * output_size + column;
                const float activated = activate(projection[index] + bias[column], column / group_size);
                output[index] = activated * sigmoid(gate[index] + gate_bias[column]) + shortcut[index];
            }
        }
        return 0;
    }
};

}  // namespace momo

extern "C" int momo_neuron_group_f32(
    const float *input, const float *weights, const float *bias,
    const float *gate_weights, const float *gate_bias,
    const float *residual, const float *residual_weights, float *output,
    size_t batch, size_t input_size, size_t output_size,
    size_t residual_size, size_t group_size) {
    return momo::NeuronGroupExecutor().run(
        input, weights, bias, gate_weights, gate_bias, residual, residual_weights,
        output, batch, input_size, output_size, residual_size, group_size);
}

extern "C" const char *momo_cpp_backend_name(void) {
    return "momo-cpp-neuron-groups";
}
