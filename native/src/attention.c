#include "momo_core.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static int checked_product(size_t left, size_t right, size_t *result) {
    if (left == 0 || right == 0) {
        return MOMO_STATUS_INVALID_ARGUMENT;
    }
    if (left > SIZE_MAX / right) {
        return MOMO_STATUS_OVERFLOW;
    }
    *result = left * right;
    if (*result > SIZE_MAX / sizeof(float)) {
        return MOMO_STATUS_OVERFLOW;
    }
    return MOMO_STATUS_OK;
}

static int finite_values(const float *values, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
}

static int attention_f32(
    const float *query, const float *key, const float *value, float *output,
    size_t query_tokens, size_t cache_tokens, size_t query_heads,
    size_t key_value_heads, size_t head_size, float scale,
    int is_decode, uint64_t query_position) {
    size_t query_rows;
    size_t cache_rows;
    size_t query_count;
    size_t cache_count;
    size_t heads_per_group;
    size_t token;
    size_t query_head;
    double *accumulator;
    int status;

    if (query == NULL || key == NULL || value == NULL || output == NULL ||
        query_heads == 0 || key_value_heads == 0 ||
        query_heads % key_value_heads != 0 || !isfinite(scale) || scale <= 0.0f) {
        return MOMO_STATUS_INVALID_ARGUMENT;
    }
    status = checked_product(query_tokens, query_heads, &query_rows);
    if (status != MOMO_STATUS_OK) {
        return status;
    }
    status = checked_product(cache_tokens, key_value_heads, &cache_rows);
    if (status != MOMO_STATUS_OK) {
        return status;
    }
    status = checked_product(query_rows, head_size, &query_count);
    if (status != MOMO_STATUS_OK) {
        return status;
    }
    status = checked_product(cache_rows, head_size, &cache_count);
    if (status != MOMO_STATUS_OK) {
        return status;
    }
    if (!finite_values(query, query_count) || !finite_values(key, cache_count) ||
        !finite_values(value, cache_count)) {
        return MOMO_STATUS_NUMERIC_ERROR;
    }
    if (head_size > SIZE_MAX / sizeof(double)) {
        return MOMO_STATUS_OVERFLOW;
    }
    accumulator = (double *)calloc(head_size, sizeof(double));
    if (accumulator == NULL) {
        return MOMO_STATUS_OUT_OF_MEMORY;
    }
    heads_per_group = query_heads / key_value_heads;
    for (token = 0; token < query_tokens; ++token) {
        size_t visible_tokens = token + 1;
        if (is_decode) {
            visible_tokens = query_position >= (uint64_t)(cache_tokens - 1)
                                 ? cache_tokens
                                 : (size_t)query_position + 1;
        }
        for (query_head = 0; query_head < query_heads; ++query_head) {
            const size_t key_head = query_head / heads_per_group;
            const size_t query_offset =
                (token * query_heads + query_head) * head_size;
            double maximum = -DBL_MAX;
            double denominator = 0.0;
            size_t key_token;
            size_t dimension;
            for (dimension = 0; dimension < head_size; ++dimension) {
                accumulator[dimension] = 0.0;
            }
            for (key_token = 0; key_token < visible_tokens; ++key_token) {
                const size_t key_offset =
                    (key_token * key_value_heads + key_head) * head_size;
                double score = 0.0;
                double next_maximum;
                double old_weight;
                double new_weight;
                for (dimension = 0; dimension < head_size; ++dimension) {
                    score += (double)query[query_offset + dimension] *
                             (double)key[key_offset + dimension];
                }
                score *= (double)scale;
                if (!isfinite(score)) {
                    free(accumulator);
                    return MOMO_STATUS_NUMERIC_ERROR;
                }
                next_maximum = maximum > score ? maximum : score;
                old_weight = maximum == -DBL_MAX ? 0.0 : exp(maximum - next_maximum);
                new_weight = exp(score - next_maximum);
                denominator = denominator * old_weight + new_weight;
                for (dimension = 0; dimension < head_size; ++dimension) {
                    accumulator[dimension] = accumulator[dimension] * old_weight +
                        new_weight * (double)value[key_offset + dimension];
                }
                maximum = next_maximum;
            }
            if (!(denominator > 0.0) || !isfinite(denominator)) {
                free(accumulator);
                return MOMO_STATUS_NUMERIC_ERROR;
            }
            for (dimension = 0; dimension < head_size; ++dimension) {
                const double normalized = accumulator[dimension] / denominator;
                if (!isfinite(normalized) || fabs(normalized) > FLT_MAX) {
                    free(accumulator);
                    return MOMO_STATUS_NUMERIC_ERROR;
                }
                output[query_offset + dimension] = (float)normalized;
            }
        }
    }
    free(accumulator);
    return MOMO_STATUS_OK;
}

int momo_causal_gqa_f32(const float *query, const float *key,
                        const float *value, float *output, size_t tokens,
                        size_t query_heads, size_t key_value_heads,
                        size_t head_size, float scale) {
    return attention_f32(query, key, value, output, tokens, tokens,
                         query_heads, key_value_heads, head_size, scale, 0, 0);
}

int momo_decode_attention_f32(
    const float *query, const float *key_cache, const float *value_cache,
    float *output, size_t cache_tokens, size_t query_heads,
    size_t key_value_heads, size_t head_size, uint64_t query_position,
    float scale) {
    return attention_f32(query, key_cache, value_cache, output, 1, cache_tokens,
                         query_heads, key_value_heads, head_size, scale, 1,
                         query_position);
}
