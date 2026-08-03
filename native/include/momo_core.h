#ifndef MOMO_CORE_H
#define MOMO_CORE_H

#include <stddef.h>

#if defined(_WIN32) && defined(MOMO_BUILDING_DLL)
#define MOMO_API __declspec(dllexport)
#elif defined(_WIN32)
#define MOMO_API __declspec(dllimport)
#else
#define MOMO_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MOMO_CORE_ABI_VERSION 1

MOMO_API int momo_matmul_f32(const float *left, const float *right, float *output,
                             size_t rows, size_t inner, size_t columns);
MOMO_API int momo_softmax_f32(const float *input, float *output, size_t rows, size_t columns);
MOMO_API int momo_layer_norm_f32(const float *input, float *output, size_t rows,
                                 size_t columns, float epsilon);
MOMO_API int momo_neuron_group_f32(
    const float *input, const float *weights, const float *bias,
    const float *gate_weights, const float *gate_bias,
    const float *residual, const float *residual_weights, float *output,
    size_t batch, size_t input_size, size_t output_size,
    size_t residual_size, size_t group_size);
MOMO_API const char *momo_cpp_backend_name(void);

#ifdef __cplusplus
}
#endif

#endif
