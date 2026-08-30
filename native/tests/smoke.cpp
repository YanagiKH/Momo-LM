#include "momo_core.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

bool close(float left, float right, float tolerance = 1e-5f) {
    return std::fabs(left - right) <= tolerance;
}

template <std::size_t Size>
bool finite(const std::array<float, Size> &values) {
    for (float value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    const std::array<float, 4> left{1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 4> right{5.0f, 6.0f, 7.0f, 8.0f};
    std::array<float, 4> output{};
    if (momo_matmul_f32(left.data(), right.data(), output.data(), 2, 2, 2) !=
        MOMO_STATUS_OK) {
        std::cerr << "matmul returned an error\n";
        return 1;
    }
    const std::array<float, 4> expected{19.0f, 22.0f, 43.0f, 50.0f};
    for (std::size_t index = 0; index < output.size(); ++index) {
        if (!close(output[index], expected[index])) {
            std::cerr << "matmul output mismatch\n";
            return 2;
        }
    }

    std::array<float, 4> probabilities{};
    if (momo_softmax_f32(output.data(), probabilities.data(), 2, 2) !=
            MOMO_STATUS_OK ||
        !close(probabilities[0] + probabilities[1], 1.0f) ||
        !close(probabilities[2] + probabilities[3], 1.0f)) {
        std::cerr << "softmax output mismatch\n";
        return 3;
    }
    std::array<float, 4> normalized{};
    const std::array<float, 2> norm_weight{1.0f, 0.5f};
    if (momo_rms_norm_f32(output.data(), norm_weight.data(), normalized.data(), 2, 2,
                          1e-5f) != MOMO_STATUS_OK ||
        !finite(normalized)) {
        std::cerr << "RMSNorm output mismatch\n";
        return 4;
    }

    const std::array<float, 8> query{1.0f, 0.0f, 0.0f, 1.0f,
                                     1.0f, 0.0f, 0.0f, 1.0f};
    const std::array<float, 4> key{1.0f, 0.0f, 0.0f, 1.0f};
    const std::array<std::uint64_t, 2> positions{0, UINT64_MAX - 4};
    std::array<float, 8> rotated_query{};
    std::array<float, 4> rotated_key{};
    if (momo_rope_f32(query.data(), key.data(), positions.data(),
                      rotated_query.data(), rotated_key.data(), 2, 2, 1, 2, 2,
                      10000.0f) != MOMO_STATUS_OK ||
        !finite(rotated_query) || !finite(rotated_key)) {
        std::cerr << "RoPE high-position output mismatch\n";
        return 5;
    }

    const std::array<float, 4> attention_query{1.0f, 0.0f, 1.0f, 0.0f};
    const std::array<float, 4> attention_key{1.0f, 0.0f, 0.0f, 1.0f};
    const std::array<float, 4> attention_value{3.0f, 4.0f, 9.0f, 10.0f};
    std::array<float, 4> attention_output{};
    if (momo_causal_gqa_f32(attention_query.data(), attention_key.data(),
                            attention_value.data(), attention_output.data(), 2, 1, 1,
                            2, 1.0f) != MOMO_STATUS_OK ||
        !close(attention_output[0], 3.0f) || !close(attention_output[1], 4.0f)) {
        std::cerr << "causal attention output mismatch\n";
        return 6;
    }
    std::array<float, 2> decode_output{};
    if (momo_decode_attention_f32(attention_query.data(), attention_key.data(),
                                  attention_value.data(), decode_output.data(), 2, 1, 1,
                                  2, UINT64_MAX, 1.0f) != MOMO_STATUS_OK ||
        !finite(decode_output)) {
        std::cerr << "decode attention output mismatch\n";
        return 7;
    }

    const std::array<float, 4> quantization_input{
        0.0f, 3.0f, -2.0f, std::numeric_limits<float>::denorm_min()};
    std::array<std::int8_t, 4> quantized{};
    std::array<float, 2> scales{};
    std::array<float, 4> dequantized{};
    if (momo_quantize_q8_f32(quantization_input.data(), quantized.data(), scales.data(),
                             2, 2) != MOMO_STATUS_OK ||
        scales[0] < std::numeric_limits<float>::min() ||
        scales[1] < std::numeric_limits<float>::min() ||
        momo_dequantize_q8_f32(quantized.data(), scales.data(), dequantized.data(), 2,
                               2) != MOMO_STATUS_OK ||
        !finite(dequantized)) {
        std::cerr << "Q8 output mismatch\n";
        return 8;
    }

    const std::array<float, 4> logits{0.0f, 1.0f, 2.0f, 3.0f};
    std::size_t first_sample = 0;
    std::size_t second_sample = 0;
    if (momo_sample_f32(logits.data(), logits.size(), 0.8f, 3, 0.9f, 7, 2,
                        &first_sample) != MOMO_STATUS_OK ||
        momo_sample_f32(logits.data(), logits.size(), 0.8f, 3, 0.9f, 7, 2,
                        &second_sample) != MOMO_STATUS_OK ||
        first_sample != second_sample || first_sample >= logits.size()) {
        std::cerr << "deterministic sampler output mismatch\n";
        return 9;
    }

    std::array<float, 2> invalid{0.0f, std::numeric_limits<float>::infinity()};
    std::array<float, 2> invalid_output{};
    if (momo_softmax_f32(invalid.data(), invalid_output.data(), 1, 2) !=
        MOMO_STATUS_NUMERIC_ERROR) {
        std::cerr << "non-finite input was not rejected\n";
        return 10;
    }
    if (momo_matmul_f32(left.data(), right.data(), output.data(),
                        std::numeric_limits<std::size_t>::max(), 2, 2) !=
        MOMO_STATUS_OVERFLOW) {
        std::cerr << "overflowing tensor shape was not rejected\n";
        return 11;
    }

    std::cout << momo_cpp_backend_name() << " ABI " << MOMO_CORE_ABI_VERSION
              << " passed\n";
    return 0;
}
