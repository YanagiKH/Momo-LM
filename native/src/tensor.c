#include "momo_core.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static int checked_product(size_t left, size_t right, size_t *result) {
    if (left == 0 || right == 0) {
        return 0;
    }
    if (left > SIZE_MAX / right) {
        return 0;
    }
    *result = left * right;
    return *result <= SIZE_MAX / sizeof(float);
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

static double rope_angle(uint64_t position, size_t dimension,
                         size_t rotary_dimensions, float theta) {
    static const unsigned int shifts[] = {48U, 32U, 16U, 0U};
    const double tau = 6.283185307179586476925286766559005768;
    const double exponent = -(double)dimension / (double)rotary_dimensions;
    const double frequency = pow((double)theta, exponent);
    double phase = 0.0;
    size_t index;
    for (index = 0; index < sizeof(shifts) / sizeof(shifts[0]); ++index) {
        const uint64_t chunk = (position >> shifts[index]) & UINT64_C(0xffff);
        phase = fmod(phase * 65536.0 + (double)chunk * frequency, tau);
    }
    return phase;
}

int momo_matmul_f32(const float *left, const float *right, float *output,
                    size_t rows, size_t inner, size_t columns) {
    const size_t tile = 32;
    size_t left_count;
    size_t right_count;
    size_t output_count;
    size_t row_block;
    size_t inner_block;
    size_t row;
    size_t depth;
    size_t column;

    if (left == NULL || right == NULL || output == NULL) {
        return MOMO_STATUS_INVALID_ARGUMENT;
    }
    if (!checked_product(rows, inner, &left_count) ||
        !checked_product(inner, columns, &right_count) ||
        !checked_product(rows, columns, &output_count)) {
        return rows == 0 || inner == 0 || columns == 0
                   ? MOMO_STATUS_INVALID_ARGUMENT
                   : MOMO_STATUS_OVERFLOW;
    }
    if (!finite_values(left, left_count) || !finite_values(right, right_count)) {
        return MOMO_STATUS_NUMERIC_ERROR;
    }
    memset(output, 0, output_count * sizeof(float));
    for (row_block = 0; row_block < rows; row_block += tile) {
        const size_t row_end = rows - row_block < tile ? rows : row_block + tile;
        for (inner_block = 0; inner_block < inner; inner_block += tile) {
            const size_t inner_end = inner - inner_block < tile ? inner : inner_block + tile;
            for (row = row_block; row < row_end; ++row) {
                float *output_row = output + row * columns;
                for (depth = inner_block; depth < inner_end; ++depth) {
                    const float value = left[row * inner + depth];
                    const float *right_row = right + depth * columns;
                    for (column = 0; column < columns; ++column) {
                        output_row[column] += value * right_row[column];
                    }
                }
            }
        }
    }
    return finite_values(output, output_count) ? MOMO_STATUS_OK : MOMO_STATUS_NUMERIC_ERROR;
}

int momo_softmax_f32(const float *input, float *output, size_t rows, size_t columns) {
    size_t count;
    size_t row;
    size_t column;
    if (input == NULL || output == NULL) {
        return MOMO_STATUS_INVALID_ARGUMENT;
    }
    if (!checked_product(rows, columns, &count)) {
        return rows == 0 || columns == 0 ? MOMO_STATUS_INVALID_ARGUMENT : MOMO_STATUS_OVERFLOW;
    }
    if (!finite_values(input, count)) {
        return MOMO_STATUS_NUMERIC_ERROR;
    }
    for (row = 0; row < rows; ++row) {
        const float *input_row = input + row * columns;
        float *output_row = output + row * columns;
        float maximum = -FLT_MAX;
        double total = 0.0;
        for (column = 0; column < columns; ++column) {
            if (input_row[column] > maximum) {
                maximum = input_row[column];
            }
        }
        for (column = 0; column < columns; ++column) {
            output_row[column] = expf(input_row[column] - maximum);
            total += output_row[column];
        }
        if (!(total > 0.0) || !isfinite(total)) {
            return MOMO_STATUS_NUMERIC_ERROR;
        }
        for (column = 0; column < columns; ++column) {
            output_row[column] = (float)(output_row[column] / total);
        }
    }
    return MOMO_STATUS_OK;
}

int momo_layer_norm_f32(const float *input, float *output, size_t rows,
                        size_t columns, float epsilon) {
    size_t count;
    size_t row;
    size_t column;
    if (input == NULL || output == NULL || !isfinite(epsilon) || epsilon <= 0.0f) {
        return MOMO_STATUS_INVALID_ARGUMENT;
    }
    if (!checked_product(rows, columns, &count)) {
        return rows == 0 || columns == 0 ? MOMO_STATUS_INVALID_ARGUMENT : MOMO_STATUS_OVERFLOW;
    }
    if (!finite_values(input, count)) {
        return MOMO_STATUS_NUMERIC_ERROR;
    }
    for (row = 0; row < rows; ++row) {
        const float *input_row = input + row * columns;
        float *output_row = output + row * columns;
        double mean = 0.0;
        double variance = 0.0;
        for (column = 0; column < columns; ++column) {
            mean += input_row[column];
        }
        mean /= (double)columns;
        for (column = 0; column < columns; ++column) {
            const double centered = (double)input_row[column] - mean;
            variance += centered * centered;
        }
        variance /= (double)columns;
        if (!isfinite(mean) || !isfinite(variance)) {
            return MOMO_STATUS_NUMERIC_ERROR;
        }
        {
            const double scale = 1.0 / sqrt(variance + (double)epsilon);
            for (column = 0; column < columns; ++column) {
                output_row[column] = (float)(((double)input_row[column] - mean) * scale);
            }
        }
    }
    return finite_values(output, count) ? MOMO_STATUS_OK : MOMO_STATUS_NUMERIC_ERROR;
}

int momo_rms_norm_f32(const float *input, const float *weight, float *output,
                      size_t rows, size_t columns, float epsilon) {
    size_t count;
    size_t row;
    size_t column;
    if (input == NULL || output == NULL || !isfinite(epsilon) || epsilon <= 0.0f) {
        return MOMO_STATUS_INVALID_ARGUMENT;
    }
    if (!checked_product(rows, columns, &count)) {
        return rows == 0 || columns == 0 ? MOMO_STATUS_INVALID_ARGUMENT : MOMO_STATUS_OVERFLOW;
    }
    if (!finite_values(input, count) || (weight != NULL && !finite_values(weight, columns))) {
        return MOMO_STATUS_NUMERIC_ERROR;
    }
    for (row = 0; row < rows; ++row) {
        const float *input_row = input + row * columns;
        float *output_row = output + row * columns;
        double mean_square = 0.0;
        for (column = 0; column < columns; ++column) {
            const double value = input_row[column];
            mean_square += value * value;
        }
        mean_square /= (double)columns;
        if (!isfinite(mean_square)) {
            return MOMO_STATUS_NUMERIC_ERROR;
        }
        {
            const double inverse_rms = 1.0 / sqrt(mean_square + (double)epsilon);
            for (column = 0; column < columns; ++column) {
                const double multiplier = weight == NULL ? 1.0 : weight[column];
                output_row[column] = (float)((double)input_row[column] * inverse_rms * multiplier);
            }
        }
    }
    return finite_values(output, count) ? MOMO_STATUS_OK : MOMO_STATUS_NUMERIC_ERROR;
}

int momo_rope_f32(const float *query, const float *key, const uint64_t *positions,
                  float *query_output, float *key_output, size_t tokens,
                  size_t query_heads, size_t key_value_heads, size_t head_size,
                  size_t rotary_dimensions, float theta) {
    size_t query_rows;
    size_t key_rows;
    size_t query_count;
    size_t key_count;
    size_t token;
    size_t head;
    size_t dimension;
    if (query == NULL || key == NULL || positions == NULL || query_output == NULL ||
        key_output == NULL || tokens == 0 || query_heads == 0 ||
        key_value_heads == 0 || head_size == 0 || rotary_dimensions == 0 ||
        rotary_dimensions > head_size || rotary_dimensions % 2 != 0 ||
        !isfinite(theta) || theta <= 0.0f) {
        return MOMO_STATUS_INVALID_ARGUMENT;
    }
    if (!checked_product(tokens, query_heads, &query_rows) ||
        !checked_product(tokens, key_value_heads, &key_rows) ||
        !checked_product(query_rows, head_size, &query_count) ||
        !checked_product(key_rows, head_size, &key_count)) {
        return MOMO_STATUS_OVERFLOW;
    }
    if (!finite_values(query, query_count) || !finite_values(key, key_count)) {
        return MOMO_STATUS_NUMERIC_ERROR;
    }
    memmove(query_output, query, query_count * sizeof(float));
    memmove(key_output, key, key_count * sizeof(float));
    for (token = 0; token < tokens; ++token) {
        for (dimension = 0; dimension < rotary_dimensions; dimension += 2) {
            const double angle = rope_angle(positions[token], dimension,
                                            rotary_dimensions, theta);
            const double cosine = cos(angle);
            const double sine = sin(angle);
            for (head = 0; head < query_heads; ++head) {
                const size_t offset = (token * query_heads + head) * head_size + dimension;
                const double first = query[offset];
                const double second = query[offset + 1];
                query_output[offset] = (float)(first * cosine - second * sine);
                query_output[offset + 1] = (float)(first * sine + second * cosine);
            }
            for (head = 0; head < key_value_heads; ++head) {
                const size_t offset = (token * key_value_heads + head) * head_size + dimension;
                const double first = key[offset];
                const double second = key[offset + 1];
                key_output[offset] = (float)(first * cosine - second * sine);
                key_output[offset + 1] = (float)(first * sine + second * cosine);
            }
        }
    }
    return finite_values(query_output, query_count) && finite_values(key_output, key_count)
               ? MOMO_STATUS_OK
               : MOMO_STATUS_NUMERIC_ERROR;
}

int momo_quantize_q8_f32(const float *input, int8_t *output, float *scales,
                         size_t rows, size_t columns) {
    size_t count;
    size_t row;
    size_t column;
    if (input == NULL || output == NULL || scales == NULL) {
        return MOMO_STATUS_INVALID_ARGUMENT;
    }
    if (!checked_product(rows, columns, &count)) {
        return rows == 0 || columns == 0 ? MOMO_STATUS_INVALID_ARGUMENT : MOMO_STATUS_OVERFLOW;
    }
    if (!finite_values(input, count)) {
        return MOMO_STATUS_NUMERIC_ERROR;
    }
    for (row = 0; row < rows; ++row) {
        const size_t offset = row * columns;
        float maximum = 0.0f;
        float scale;
        for (column = 0; column < columns; ++column) {
            const float magnitude = fabsf(input[offset + column]);
            if (magnitude > maximum) {
                maximum = magnitude;
            }
        }
        if (maximum == 0.0f) {
            scale = 1.0f;
        } else {
            scale = maximum / 127.0f;
            if (scale < FLT_MIN) {
                scale = FLT_MIN;
            }
        }
        scales[row] = scale;
        for (column = 0; column < columns; ++column) {
            long quantized = (long)roundf(input[offset + column] / scale);
            if (quantized > 127) {
                quantized = 127;
            } else if (quantized < -127) {
                quantized = -127;
            }
            output[offset + column] = (int8_t)quantized;
        }
    }
    return MOMO_STATUS_OK;
}

int momo_dequantize_q8_f32(const int8_t *input, const float *scales,
                           float *output, size_t rows, size_t columns) {
    size_t count;
    size_t row;
    size_t column;
    if (input == NULL || scales == NULL || output == NULL) {
        return MOMO_STATUS_INVALID_ARGUMENT;
    }
    if (!checked_product(rows, columns, &count)) {
        return rows == 0 || columns == 0 ? MOMO_STATUS_INVALID_ARGUMENT : MOMO_STATUS_OVERFLOW;
    }
    for (row = 0; row < rows; ++row) {
        const float scale = scales[row];
        const size_t offset = row * columns;
        if (!isfinite(scale) || scale < FLT_MIN) {
            return MOMO_STATUS_NUMERIC_ERROR;
        }
        for (column = 0; column < columns; ++column) {
            output[offset + column] = (float)input[offset + column] * scale;
        }
    }
    return finite_values(output, count) ? MOMO_STATUS_OK : MOMO_STATUS_NUMERIC_ERROR;
}
