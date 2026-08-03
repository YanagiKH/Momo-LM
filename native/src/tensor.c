#include "momo_core.h"

#include <float.h>
#include <math.h>
#include <string.h>

int momo_matmul_f32(const float *left, const float *right, float *output,
                    size_t rows, size_t inner, size_t columns) {
    const size_t tile = 32;
    size_t row_block;
    size_t inner_block;
    size_t row;
    size_t depth;
    size_t column;

    if (left == NULL || right == NULL || output == NULL) {
        return -1;
    }
    if (rows == 0 || inner == 0 || columns == 0) {
        return -2;
    }
    memset(output, 0, rows * columns * sizeof(float));
    for (row_block = 0; row_block < rows; row_block += tile) {
        const size_t row_end = row_block + tile < rows ? row_block + tile : rows;
        for (inner_block = 0; inner_block < inner; inner_block += tile) {
            const size_t inner_end = inner_block + tile < inner ? inner_block + tile : inner;
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
    return 0;
}

int momo_softmax_f32(const float *input, float *output, size_t rows, size_t columns) {
    size_t row;
    size_t column;
    if (input == NULL || output == NULL || rows == 0 || columns == 0) {
        return -1;
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
        if (total <= 0.0) {
            return -2;
        }
        for (column = 0; column < columns; ++column) {
            output_row[column] = (float)(output_row[column] / total);
        }
    }
    return 0;
}

int momo_layer_norm_f32(const float *input, float *output, size_t rows,
                        size_t columns, float epsilon) {
    size_t row;
    size_t column;
    if (input == NULL || output == NULL || rows == 0 || columns == 0 || epsilon <= 0.0f) {
        return -1;
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
            const double centered = input_row[column] - mean;
            variance += centered * centered;
        }
        variance /= (double)columns;
        const float scale = 1.0f / sqrtf((float)variance + epsilon);
        for (column = 0; column < columns; ++column) {
            output_row[column] = (input_row[column] - (float)mean) * scale;
        }
    }
    return 0;
}
