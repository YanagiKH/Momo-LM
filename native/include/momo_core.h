#ifndef MOMO_CORE_H
#define MOMO_CORE_H

#include <stddef.h>
#include <stdint.h>

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

#define MOMO_CORE_ABI_VERSION 2

enum momo_status {
    MOMO_STATUS_OK = 0,
    MOMO_STATUS_INVALID_ARGUMENT = -1,
    MOMO_STATUS_NUMERIC_ERROR = -2,
    MOMO_STATUS_OVERFLOW = -3,
    MOMO_STATUS_OUT_OF_MEMORY = -4
};

MOMO_API int momo_matmul_f32(const float *left, const float *right, float *output,
                             size_t rows, size_t inner, size_t columns);
MOMO_API int momo_softmax_f32(const float *input, float *output, size_t rows, size_t columns);
MOMO_API int momo_layer_norm_f32(const float *input, float *output, size_t rows,
                                 size_t columns, float epsilon);
MOMO_API int momo_rms_norm_f32(const float *input, const float *weight,
                               float *output, size_t rows, size_t columns,
                               float epsilon);
MOMO_API int momo_rope_f32(
    const float *query, const float *key, const uint64_t *positions,
    float *query_output, float *key_output, size_t tokens,
    size_t query_heads, size_t key_value_heads, size_t head_size,
    size_t rotary_dimensions, float theta);
MOMO_API int momo_causal_gqa_f32(
    const float *query, const float *key, const float *value, float *output,
    size_t tokens, size_t query_heads, size_t key_value_heads,
    size_t head_size, float scale);
MOMO_API int momo_decode_attention_f32(
    const float *query, const float *key_cache, const float *value_cache,
    float *output, size_t cache_tokens, size_t query_heads,
    size_t key_value_heads, size_t head_size, uint64_t query_position,
    float scale);
MOMO_API int momo_quantize_q8_f32(const float *input, int8_t *output,
                                  float *scales, size_t rows, size_t columns);
MOMO_API int momo_dequantize_q8_f32(const int8_t *input, const float *scales,
                                    float *output, size_t rows, size_t columns);
MOMO_API int momo_sample_f32(const float *logits, size_t count,
                             float temperature, size_t top_k, float top_p,
                             uint64_t seed, uint64_t counter,
                             size_t *sampled_index);
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
