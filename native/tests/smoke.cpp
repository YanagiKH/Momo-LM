#include "momo_core.h"

#include <array>
#include <cmath>
#include <iostream>

int main() {
    const std::array<float, 4> left{1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 4> right{5.0f, 6.0f, 7.0f, 8.0f};
    std::array<float, 4> output{};
    if (momo_matmul_f32(left.data(), right.data(), output.data(), 2, 2, 2) != 0) {
        std::cerr << "matmul returned an error\n";
        return 1;
    }
    const std::array<float, 4> expected{19.0f, 22.0f, 43.0f, 50.0f};
    for (std::size_t index = 0; index < output.size(); ++index) {
        if (std::fabs(output[index] - expected[index]) > 1e-5f) {
            std::cerr << "matmul output mismatch\n";
            return 2;
        }
    }
    std::array<float, 4> probabilities{};
    if (momo_softmax_f32(output.data(), probabilities.data(), 2, 2) != 0 ||
        std::fabs(probabilities[0] + probabilities[1] - 1.0f) > 1e-5f ||
        std::fabs(probabilities[2] + probabilities[3] - 1.0f) > 1e-5f) {
        std::cerr << "softmax output mismatch\n";
        return 3;
    }
    std::cout << momo_cpp_backend_name() << " ABI " << MOMO_CORE_ABI_VERSION << " passed\n";
    return 0;
}
