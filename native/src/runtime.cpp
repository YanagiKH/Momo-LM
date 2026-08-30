#include "momo_core.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace momo {

bool checked_product(std::size_t left, std::size_t right, std::size_t &result) {
    if (left == 0 || right == 0 || left > std::numeric_limits<std::size_t>::max() / right) {
        return false;
    }
    result = left * right;
    return result <= std::numeric_limits<std::size_t>::max() / sizeof(float);
}

bool finite_values(const float *values, std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) {
            return false;
        }
    }
    return true;
}

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
            return 0.5f * value *
                   (1.0f + std::tanh(coefficient * (value + 0.044715f * cubic)));
        }
        default:
            return value * sigmoid(value);
    }
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

double uniform_sample(std::uint64_t seed, std::uint64_t counter) {
    const std::uint64_t mixed = splitmix64(seed +
        UINT64_C(0x9e3779b97f4a7c15) * (counter + UINT64_C(1)));
    return static_cast<double>(mixed >> 11U) * (1.0 / 9007199254740992.0);
}

class NeuronGroupExecutor {
  public:
    int run(const float *input, const float *weights, const float *bias,
            const float *gate_weights, const float *gate_bias,
            const float *residual, const float *residual_weights, float *output,
            std::size_t batch, std::size_t input_size, std::size_t output_size,
            std::size_t residual_size, std::size_t group_size) const {
        std::size_t input_count;
        std::size_t weight_count;
        std::size_t output_count;
        std::size_t residual_count;
        std::size_t residual_weight_count;
        if (input == nullptr || weights == nullptr || bias == nullptr ||
            gate_weights == nullptr || gate_bias == nullptr || residual == nullptr ||
            residual_weights == nullptr || output == nullptr || group_size == 0 ||
            !checked_product(batch, input_size, input_count) ||
            !checked_product(input_size, output_size, weight_count) ||
            !checked_product(batch, output_size, output_count) ||
            !checked_product(batch, residual_size, residual_count) ||
            !checked_product(residual_size, output_size, residual_weight_count)) {
            return MOMO_STATUS_INVALID_ARGUMENT;
        }
        if (!finite_values(input, input_count) || !finite_values(weights, weight_count) ||
            !finite_values(bias, output_size) ||
            !finite_values(gate_weights, weight_count) ||
            !finite_values(gate_bias, output_size) ||
            !finite_values(residual, residual_count) ||
            !finite_values(residual_weights, residual_weight_count)) {
            return MOMO_STATUS_NUMERIC_ERROR;
        }
        try {
            std::vector<float> projection(output_count);
            std::vector<float> gate(output_count);
            std::vector<float> shortcut(output_count);
            if (momo_matmul_f32(input, weights, projection.data(), batch, input_size,
                                output_size) != MOMO_STATUS_OK ||
                momo_matmul_f32(input, gate_weights, gate.data(), batch, input_size,
                                output_size) != MOMO_STATUS_OK ||
                momo_matmul_f32(residual, residual_weights, shortcut.data(), batch,
                                residual_size, output_size) != MOMO_STATUS_OK) {
                return MOMO_STATUS_NUMERIC_ERROR;
            }
            for (std::size_t row = 0; row < batch; ++row) {
                for (std::size_t column = 0; column < output_size; ++column) {
                    const std::size_t index = row * output_size + column;
                    const float projected = projection[index] + bias[column];
                    const float gated = gate[index] + gate_bias[column];
                    if (!std::isfinite(projected) || !std::isfinite(gated)) {
                        return MOMO_STATUS_NUMERIC_ERROR;
                    }
                    const float activated = activate(projected, column / group_size);
                    output[index] = activated * sigmoid(gated) + shortcut[index];
                    if (!std::isfinite(output[index])) {
                        return MOMO_STATUS_NUMERIC_ERROR;
                    }
                }
            }
        } catch (const std::bad_alloc &) {
            return MOMO_STATUS_OUT_OF_MEMORY;
        } catch (...) {
            return MOMO_STATUS_NUMERIC_ERROR;
        }
        return MOMO_STATUS_OK;
    }
};

}  // namespace momo

extern "C" int momo_sample_f32(const float *logits, size_t count,
                               float temperature, size_t top_k, float top_p,
                               uint64_t seed, uint64_t counter,
                               size_t *sampled_index) {
    struct Candidate {
        float logit;
        std::size_t index;
        double weight;
    };
    if (logits == nullptr || sampled_index == nullptr || count == 0 ||
        !std::isfinite(temperature) || temperature < 0.0f ||
        !std::isfinite(top_p) || top_p <= 0.0f || top_p > 1.0f ||
        top_k > count) {
        return MOMO_STATUS_INVALID_ARGUMENT;
    }
    if (!momo::finite_values(logits, count)) {
        return MOMO_STATUS_NUMERIC_ERROR;
    }
    if (temperature == 0.0f) {
        *sampled_index = static_cast<std::size_t>(
            std::max_element(logits, logits + count) - logits);
        return MOMO_STATUS_OK;
    }
    try {
        std::vector<Candidate> candidates;
        candidates.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            candidates.push_back({logits[index] / temperature, index, 0.0});
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate &left, const Candidate &right) {
                      return left.logit > right.logit ||
                             (left.logit == right.logit && left.index < right.index);
                  });
        if (top_k != 0) {
            candidates.resize(top_k);
        }
        const double maximum = candidates.front().logit;
        double total = 0.0;
        for (Candidate &candidate : candidates) {
            candidate.weight = std::exp(static_cast<double>(candidate.logit) - maximum);
            total += candidate.weight;
        }
        if (!(total > 0.0) || !std::isfinite(total)) {
            return MOMO_STATUS_NUMERIC_ERROR;
        }
        double kept_mass = 0.0;
        std::size_t kept = 0;
        do {
            kept_mass += candidates[kept].weight;
            ++kept;
        } while (kept < candidates.size() && kept_mass / total < top_p);
        const double target = momo::uniform_sample(seed, counter) * kept_mass;
        double cumulative = 0.0;
        for (std::size_t index = 0; index < kept; ++index) {
            cumulative += candidates[index].weight;
            if (target < cumulative || index + 1 == kept) {
                *sampled_index = candidates[index].index;
                return MOMO_STATUS_OK;
            }
        }
    } catch (const std::bad_alloc &) {
        return MOMO_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return MOMO_STATUS_NUMERIC_ERROR;
    }
    return MOMO_STATUS_NUMERIC_ERROR;
}

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
    return "momo-cpp-abi2";
}
