use std::ffi::c_char;
use std::slice;

pub const ABI_VERSION: i32 = 1;
static BACKEND_NAME: &[u8] = b"momo-rust-safe-kernels\0";

fn dimensions_are_valid(rows: usize, inner: usize, columns: usize) -> bool {
    rows > 0
        && inner > 0
        && columns > 0
        && rows.checked_mul(inner).is_some()
        && inner.checked_mul(columns).is_some()
        && rows.checked_mul(columns).is_some()
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn activate(value: f32, group: usize) -> f32 {
    match group % 3 {
        0 => value.tanh(),
        1 => {
            let transformed = 0.797_884_6 * (value + 0.044_715 * value.powi(3));
            0.5 * value * (1.0 + transformed.tanh())
        }
        _ => value * sigmoid(value),
    }
}

fn matmul(left: &[f32], right: &[f32], output: &mut [f32], rows: usize, inner: usize, columns: usize) {
    output.fill(0.0);
    const TILE: usize = 32;
    for row_block in (0..rows).step_by(TILE) {
        let row_end = (row_block + TILE).min(rows);
        for inner_block in (0..inner).step_by(TILE) {
            let inner_end = (inner_block + TILE).min(inner);
            for row in row_block..row_end {
                for depth in inner_block..inner_end {
                    let value = left[row * inner + depth];
                    for column in 0..columns {
                        output[row * columns + column] += value * right[depth * columns + column];
                    }
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn momo_rust_abi_version() -> i32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn momo_rust_backend_name() -> *const c_char {
    BACKEND_NAME.as_ptr().cast()
}

/// Multiplies two row-major float32 matrices into a caller-owned output buffer.
///
/// # Safety
///
/// Every pointer must be non-null, aligned and valid for the element count implied by
/// `rows`, `inner` and `columns`. Input and output memory must not overlap.
#[no_mangle]
pub unsafe extern "C" fn momo_rust_matmul_f32(
    left: *const f32,
    right: *const f32,
    output: *mut f32,
    rows: usize,
    inner: usize,
    columns: usize,
) -> i32 {
    if left.is_null() || right.is_null() || output.is_null() || !dimensions_are_valid(rows, inner, columns) {
        return -1;
    }
    let left = slice::from_raw_parts(left, rows * inner);
    let right = slice::from_raw_parts(right, inner * columns);
    let output = slice::from_raw_parts_mut(output, rows * columns);
    matmul(left, right, output, rows, inner, columns);
    0
}

/// Applies stable softmax independently to every row.
///
/// # Safety
///
/// `input` and `output` must each be valid for `rows * columns` float32 values and
/// must not overlap.
#[no_mangle]
pub unsafe extern "C" fn momo_rust_softmax_f32(
    input: *const f32,
    output: *mut f32,
    rows: usize,
    columns: usize,
) -> i32 {
    if input.is_null() || output.is_null() || rows == 0 || columns == 0 || rows.checked_mul(columns).is_none() {
        return -1;
    }
    let input = slice::from_raw_parts(input, rows * columns);
    let output = slice::from_raw_parts_mut(output, rows * columns);
    for row in 0..rows {
        let offset = row * columns;
        let values = &input[offset..offset + columns];
        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0_f64;
        for column in 0..columns {
            output[offset + column] = (values[column] - maximum).exp();
            total += f64::from(output[offset + column]);
        }
        if total <= 0.0 {
            return -2;
        }
        for column in 0..columns {
            output[offset + column] = (f64::from(output[offset + column]) / total) as f32;
        }
    }
    0
}

/// Normalizes every input row using its mean and variance.
///
/// # Safety
///
/// `input` and `output` must each be valid for `rows * columns` float32 values and
/// must not overlap.
#[no_mangle]
pub unsafe extern "C" fn momo_rust_layer_norm_f32(
    input: *const f32,
    output: *mut f32,
    rows: usize,
    columns: usize,
    epsilon: f32,
) -> i32 {
    if input.is_null() || output.is_null() || rows == 0 || columns == 0 || epsilon <= 0.0 {
        return -1;
    }
    let input = slice::from_raw_parts(input, rows * columns);
    let output = slice::from_raw_parts_mut(output, rows * columns);
    for row in 0..rows {
        let offset = row * columns;
        let values = &input[offset..offset + columns];
        let mean = values.iter().map(|&value| f64::from(value)).sum::<f64>() / columns as f64;
        let variance = values
            .iter()
            .map(|&value| {
                let centered = f64::from(value) - mean;
                centered * centered
            })
            .sum::<f64>()
            / columns as f64;
        let scale = 1.0 / (variance as f32 + epsilon).sqrt();
        for column in 0..columns {
            output[offset + column] = (values[column] - mean as f32) * scale;
        }
    }
    0
}

/// Executes projection, gating, mixed activation and a residual projection.
///
/// # Safety
///
/// All pointers must be non-null, aligned, non-overlapping and valid for the matrix
/// dimensions supplied to this function.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn momo_rust_neuron_group_f32(
    input: *const f32,
    weights: *const f32,
    bias: *const f32,
    gate_weights: *const f32,
    gate_bias: *const f32,
    residual: *const f32,
    residual_weights: *const f32,
    output: *mut f32,
    batch: usize,
    input_size: usize,
    output_size: usize,
    residual_size: usize,
    group_size: usize,
) -> i32 {
    if input.is_null()
        || weights.is_null()
        || bias.is_null()
        || gate_weights.is_null()
        || gate_bias.is_null()
        || residual.is_null()
        || residual_weights.is_null()
        || output.is_null()
        || group_size == 0
        || !dimensions_are_valid(batch, input_size, output_size)
        || !dimensions_are_valid(batch, residual_size, output_size)
    {
        return -1;
    }
    let input = slice::from_raw_parts(input, batch * input_size);
    let weights = slice::from_raw_parts(weights, input_size * output_size);
    let bias = slice::from_raw_parts(bias, output_size);
    let gate_weights = slice::from_raw_parts(gate_weights, input_size * output_size);
    let gate_bias = slice::from_raw_parts(gate_bias, output_size);
    let residual = slice::from_raw_parts(residual, batch * residual_size);
    let residual_weights = slice::from_raw_parts(residual_weights, residual_size * output_size);
    let output = slice::from_raw_parts_mut(output, batch * output_size);
    let mut projection = vec![0.0; batch * output_size];
    let mut gate = vec![0.0; batch * output_size];
    let mut shortcut = vec![0.0; batch * output_size];
    matmul(input, weights, &mut projection, batch, input_size, output_size);
    matmul(input, gate_weights, &mut gate, batch, input_size, output_size);
    matmul(residual, residual_weights, &mut shortcut, batch, residual_size, output_size);
    for row in 0..batch {
        for column in 0..output_size {
            let index = row * output_size + column;
            output[index] = activate(projection[index] + bias[column], column / group_size)
                * sigmoid(gate[index] + gate_bias[column])
                + shortcut[index];
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_multiplication_is_correct() {
        let left = [1.0, 2.0, 3.0, 4.0];
        let right = [5.0, 6.0, 7.0, 8.0];
        let mut output = [0.0; 4];
        let status = unsafe { momo_rust_matmul_f32(left.as_ptr(), right.as_ptr(), output.as_mut_ptr(), 2, 2, 2) };
        assert_eq!(status, 0);
        assert_eq!(output, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let input = [1.0, 2.0, 3.0, -1.0, 0.0, 1.0];
        let mut output = [0.0; 6];
        let status = unsafe { momo_rust_softmax_f32(input.as_ptr(), output.as_mut_ptr(), 2, 3) };
        assert_eq!(status, 0);
        assert!((output[0..3].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((output[3..6].iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
