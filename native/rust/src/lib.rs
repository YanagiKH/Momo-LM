use std::cmp::Ordering;
use std::ffi::c_char;
use std::mem::size_of;
use std::slice;

pub const ABI_VERSION: i32 = 2;
const OK: i32 = 0;
const INVALID_ARGUMENT: i32 = -1;
const NUMERIC_ERROR: i32 = -2;
const OVERFLOW: i32 = -3;
const OUT_OF_MEMORY: i32 = -4;
static BACKEND_NAME: &[u8] = b"momo-rust-abi2-safe-kernels\0";

fn checked_product(left: usize, right: usize) -> Option<usize> {
    if left == 0 || right == 0 {
        return None;
    }
    let count = left.checked_mul(right)?;
    if count > isize::MAX as usize / size_of::<f32>() {
        None
    } else {
        Some(count)
    }
}

fn dimensions_are_valid(rows: usize, inner: usize, columns: usize) -> bool {
    checked_product(rows, inner).is_some()
        && checked_product(inner, columns).is_some()
        && checked_product(rows, columns).is_some()
}

fn finite_values(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn allocate_f32(count: usize) -> Result<Vec<f32>, i32> {
    if count > isize::MAX as usize / size_of::<f32>() {
        return Err(OVERFLOW);
    }
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| OUT_OF_MEMORY)?;
    values.resize(count, 0.0);
    Ok(values)
}

fn allocate_f64(count: usize) -> Result<Vec<f64>, i32> {
    if count > isize::MAX as usize / size_of::<f64>() {
        return Err(OVERFLOW);
    }
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| OUT_OF_MEMORY)?;
    values.resize(count, 0.0);
    Ok(values)
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

fn matmul(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) -> i32 {
    if !finite_values(left) || !finite_values(right) {
        return NUMERIC_ERROR;
    }
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
    if finite_values(output) {
        OK
    } else {
        NUMERIC_ERROR
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn uniform_sample(seed: u64, counter: u64) -> f64 {
    let mixed = splitmix64(
        seed.wrapping_add(0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(counter.wrapping_add(1))),
    );
    ((mixed >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
}

fn rope_angle(position: u64, dimension: usize, rotary_dimensions: usize, theta: f32) -> f64 {
    let frequency = f64::from(theta).powf(-(dimension as f64) / rotary_dimensions as f64);
    let mut phase = 0.0_f64;
    for shift in [48_u32, 32, 16, 0] {
        let chunk = (position >> shift) & 0xffff;
        phase = (phase * 65_536.0 + chunk as f64 * frequency).rem_euclid(std::f64::consts::TAU);
    }
    phase
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
    if left.is_null()
        || right.is_null()
        || output.is_null()
        || !dimensions_are_valid(rows, inner, columns)
    {
        return INVALID_ARGUMENT;
    }
    let left_count = rows * inner;
    let right_count = inner * columns;
    let output_count = rows * columns;
    let left = slice::from_raw_parts(left, left_count);
    let right = slice::from_raw_parts(right, right_count);
    let output = slice::from_raw_parts_mut(output, output_count);
    matmul(left, right, output, rows, inner, columns)
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
    let count = match checked_product(rows, columns) {
        Some(count) => count,
        None => return INVALID_ARGUMENT,
    };
    if input.is_null() || output.is_null() {
        return INVALID_ARGUMENT;
    }
    let input = slice::from_raw_parts(input, count);
    let output = slice::from_raw_parts_mut(output, count);
    if !finite_values(input) {
        return NUMERIC_ERROR;
    }
    for row in 0..rows {
        let offset = row * columns;
        let values = &input[offset..offset + columns];
        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0_f64;
        for column in 0..columns {
            output[offset + column] = (values[column] - maximum).exp();
            total += f64::from(output[offset + column]);
        }
        if total <= 0.0 || !total.is_finite() {
            return NUMERIC_ERROR;
        }
        for column in 0..columns {
            output[offset + column] = (f64::from(output[offset + column]) / total) as f32;
        }
    }
    OK
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
    let count = match checked_product(rows, columns) {
        Some(count) => count,
        None => return INVALID_ARGUMENT,
    };
    if input.is_null() || output.is_null() || !epsilon.is_finite() || epsilon <= 0.0 {
        return INVALID_ARGUMENT;
    }
    let input = slice::from_raw_parts(input, count);
    let output = slice::from_raw_parts_mut(output, count);
    if !finite_values(input) {
        return NUMERIC_ERROR;
    }
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
        if !mean.is_finite() || !variance.is_finite() {
            return NUMERIC_ERROR;
        }
        let scale = 1.0 / (variance + f64::from(epsilon)).sqrt();
        for column in 0..columns {
            output[offset + column] = ((f64::from(values[column]) - mean) * scale) as f32;
        }
    }
    if finite_values(output) {
        OK
    } else {
        NUMERIC_ERROR
    }
}

/// Applies weighted RMS normalization independently to each row.
///
/// # Safety
///
/// The input and output pointers must be valid for `rows * columns` values. `weight`
/// must be valid for `columns` values. All buffers must be non-overlapping.
#[no_mangle]
pub unsafe extern "C" fn momo_rust_rms_norm_f32(
    input: *const f32,
    weight: *const f32,
    output: *mut f32,
    rows: usize,
    columns: usize,
    epsilon: f32,
) -> i32 {
    let count = match checked_product(rows, columns) {
        Some(count) => count,
        None => return INVALID_ARGUMENT,
    };
    if input.is_null()
        || weight.is_null()
        || output.is_null()
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return INVALID_ARGUMENT;
    }
    let input = slice::from_raw_parts(input, count);
    let weight = slice::from_raw_parts(weight, columns);
    let output = slice::from_raw_parts_mut(output, count);
    if !finite_values(input) || !finite_values(weight) {
        return NUMERIC_ERROR;
    }
    for row in 0..rows {
        let offset = row * columns;
        let mut mean_square = 0.0_f64;
        for column in 0..columns {
            let value = f64::from(input[offset + column]);
            mean_square += value * value;
        }
        mean_square /= columns as f64;
        if !mean_square.is_finite() {
            return NUMERIC_ERROR;
        }
        let inverse_rms = 1.0 / (mean_square + f64::from(epsilon)).sqrt();
        for column in 0..columns {
            output[offset + column] = (f64::from(input[offset + column])
                * inverse_rms
                * f64::from(weight[column])) as f32;
        }
    }
    if finite_values(output) {
        OK
    } else {
        NUMERIC_ERROR
    }
}

/// Applies rotary position embeddings to query and key tensors.
///
/// # Safety
///
/// Query and output buffers must match `tokens * heads * head_size`; positions must
/// contain `tokens` values. Buffers must be aligned and non-overlapping.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn momo_rust_rope_f32(
    query: *const f32,
    key: *const f32,
    positions: *const u64,
    query_output: *mut f32,
    key_output: *mut f32,
    tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_size: usize,
    rotary_dimensions: usize,
    theta: f32,
) -> i32 {
    let query_rows = match checked_product(tokens, query_heads) {
        Some(count) => count,
        None => return INVALID_ARGUMENT,
    };
    let key_rows = match checked_product(tokens, key_value_heads) {
        Some(count) => count,
        None => return INVALID_ARGUMENT,
    };
    let query_count = match checked_product(query_rows, head_size) {
        Some(count) => count,
        None => return OVERFLOW,
    };
    let key_count = match checked_product(key_rows, head_size) {
        Some(count) => count,
        None => return OVERFLOW,
    };
    if query.is_null()
        || key.is_null()
        || positions.is_null()
        || query_output.is_null()
        || key_output.is_null()
        || rotary_dimensions == 0
        || rotary_dimensions > head_size
        || rotary_dimensions % 2 != 0
        || !theta.is_finite()
        || theta <= 0.0
    {
        return INVALID_ARGUMENT;
    }
    let query = slice::from_raw_parts(query, query_count);
    let key = slice::from_raw_parts(key, key_count);
    let positions = slice::from_raw_parts(positions, tokens);
    let query_output = slice::from_raw_parts_mut(query_output, query_count);
    let key_output = slice::from_raw_parts_mut(key_output, key_count);
    if !finite_values(query) || !finite_values(key) {
        return NUMERIC_ERROR;
    }
    query_output.copy_from_slice(query);
    key_output.copy_from_slice(key);
    for token in 0..tokens {
        for dimension in (0..rotary_dimensions).step_by(2) {
            let angle = rope_angle(positions[token], dimension, rotary_dimensions, theta);
            let (sine, cosine) = angle.sin_cos();
            for head in 0..query_heads {
                let offset = (token * query_heads + head) * head_size + dimension;
                let first = f64::from(query[offset]);
                let second = f64::from(query[offset + 1]);
                query_output[offset] = (first * cosine - second * sine) as f32;
                query_output[offset + 1] = (first * sine + second * cosine) as f32;
            }
            for head in 0..key_value_heads {
                let offset = (token * key_value_heads + head) * head_size + dimension;
                let first = f64::from(key[offset]);
                let second = f64::from(key[offset + 1]);
                key_output[offset] = (first * cosine - second * sine) as f32;
                key_output[offset + 1] = (first * sine + second * cosine) as f32;
            }
        }
    }
    if finite_values(query_output) && finite_values(key_output) {
        OK
    } else {
        NUMERIC_ERROR
    }
}

#[allow(clippy::too_many_arguments)]
fn attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    output: &mut [f32],
    query_tokens: usize,
    cache_tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_size: usize,
    scale: f32,
    query_position: Option<u64>,
) -> i32 {
    if !finite_values(query) || !finite_values(key) || !finite_values(value) {
        return NUMERIC_ERROR;
    }
    let mut accumulator = match allocate_f64(head_size) {
        Ok(values) => values,
        Err(status) => return status,
    };
    let heads_per_group = query_heads / key_value_heads;
    for token in 0..query_tokens {
        let visible_tokens = match query_position {
            None => token + 1,
            Some(position) => {
                if position >= (cache_tokens - 1) as u64 {
                    cache_tokens
                } else {
                    position as usize + 1
                }
            }
        };
        for query_head in 0..query_heads {
            accumulator.fill(0.0);
            let key_head = query_head / heads_per_group;
            let query_offset = (token * query_heads + query_head) * head_size;
            let mut maximum = f64::NEG_INFINITY;
            let mut denominator = 0.0_f64;
            for key_token in 0..visible_tokens {
                let key_offset = (key_token * key_value_heads + key_head) * head_size;
                let mut score = 0.0_f64;
                for dimension in 0..head_size {
                    score += f64::from(query[query_offset + dimension])
                        * f64::from(key[key_offset + dimension]);
                }
                score *= f64::from(scale);
                if !score.is_finite() {
                    return NUMERIC_ERROR;
                }
                let next_maximum = maximum.max(score);
                let old_weight = if maximum.is_finite() {
                    (maximum - next_maximum).exp()
                } else {
                    0.0
                };
                let new_weight = (score - next_maximum).exp();
                denominator = denominator * old_weight + new_weight;
                for dimension in 0..head_size {
                    accumulator[dimension] = accumulator[dimension] * old_weight
                        + new_weight * f64::from(value[key_offset + dimension]);
                }
                maximum = next_maximum;
            }
            if denominator <= 0.0 || !denominator.is_finite() {
                return NUMERIC_ERROR;
            }
            for dimension in 0..head_size {
                let normalized = accumulator[dimension] / denominator;
                if !normalized.is_finite() || normalized.abs() > f64::from(f32::MAX) {
                    return NUMERIC_ERROR;
                }
                output[query_offset + dimension] = normalized as f32;
            }
        }
    }
    OK
}

/// Runs online-softmax causal grouped-query attention.
///
/// # Safety
///
/// All pointers must be valid for their tensor shapes and non-overlapping.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn momo_rust_causal_gqa_f32(
    query: *const f32,
    key: *const f32,
    value: *const f32,
    output: *mut f32,
    tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_size: usize,
    scale: f32,
) -> i32 {
    let query_rows = match checked_product(tokens, query_heads) {
        Some(count) => count,
        None => return INVALID_ARGUMENT,
    };
    let cache_rows = match checked_product(tokens, key_value_heads) {
        Some(count) => count,
        None => return INVALID_ARGUMENT,
    };
    let query_count = match checked_product(query_rows, head_size) {
        Some(count) => count,
        None => return OVERFLOW,
    };
    let cache_count = match checked_product(cache_rows, head_size) {
        Some(count) => count,
        None => return OVERFLOW,
    };
    if query.is_null()
        || key.is_null()
        || value.is_null()
        || output.is_null()
        || query_heads % key_value_heads != 0
        || !scale.is_finite()
        || scale <= 0.0
    {
        return INVALID_ARGUMENT;
    }
    attention(
        slice::from_raw_parts(query, query_count),
        slice::from_raw_parts(key, cache_count),
        slice::from_raw_parts(value, cache_count),
        slice::from_raw_parts_mut(output, query_count),
        tokens,
        tokens,
        query_heads,
        key_value_heads,
        head_size,
        scale,
        None,
    )
}

/// Runs one-token grouped-query attention against a KV cache.
///
/// # Safety
///
/// All pointers must be valid for their tensor shapes and non-overlapping.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn momo_rust_decode_attention_f32(
    query: *const f32,
    key_cache: *const f32,
    value_cache: *const f32,
    output: *mut f32,
    cache_tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_size: usize,
    query_position: u64,
    scale: f32,
) -> i32 {
    let query_count = match checked_product(query_heads, head_size) {
        Some(count) => count,
        None => return INVALID_ARGUMENT,
    };
    let cache_rows = match checked_product(cache_tokens, key_value_heads) {
        Some(count) => count,
        None => return INVALID_ARGUMENT,
    };
    let cache_count = match checked_product(cache_rows, head_size) {
        Some(count) => count,
        None => return OVERFLOW,
    };
    if query.is_null()
        || key_cache.is_null()
        || value_cache.is_null()
        || output.is_null()
        || query_heads % key_value_heads != 0
        || !scale.is_finite()
        || scale <= 0.0
    {
        return INVALID_ARGUMENT;
    }
    attention(
        slice::from_raw_parts(query, query_count),
        slice::from_raw_parts(key_cache, cache_count),
        slice::from_raw_parts(value_cache, cache_count),
        slice::from_raw_parts_mut(output, query_count),
        1,
        cache_tokens,
        query_heads,
        key_value_heads,
        head_size,
        scale,
        Some(query_position),
    )
}

/// Quantizes each input row to symmetric signed int8 values and one float32 scale.
///
/// # Safety
///
/// Buffers must be valid for the dimensions supplied and non-overlapping.
#[no_mangle]
pub unsafe extern "C" fn momo_rust_quantize_q8_f32(
    input: *const f32,
    output: *mut i8,
    scales: *mut f32,
    rows: usize,
    columns: usize,
) -> i32 {
    let count = match checked_product(rows, columns) {
        Some(count) => count,
        None => return INVALID_ARGUMENT,
    };
    if input.is_null() || output.is_null() || scales.is_null() {
        return INVALID_ARGUMENT;
    }
    let input = slice::from_raw_parts(input, count);
    let output = slice::from_raw_parts_mut(output, count);
    let scales = slice::from_raw_parts_mut(scales, rows);
    if !finite_values(input) {
        return NUMERIC_ERROR;
    }
    for row in 0..rows {
        let offset = row * columns;
        let mut maximum = 0.0_f32;
        for column in 0..columns {
            maximum = maximum.max(input[offset + column].abs());
        }
        let scale = if maximum == 0.0 {
            1.0
        } else {
            (maximum / 127.0).max(f32::MIN_POSITIVE)
        };
        scales[row] = scale;
        for column in 0..columns {
            output[offset + column] = (input[offset + column] / scale)
                .round()
                .clamp(-127.0, 127.0) as i8;
        }
    }
    OK
}

/// Dequantizes symmetric signed int8 rows using one float32 scale per row.
///
/// # Safety
///
/// Buffers must be valid for the dimensions supplied and non-overlapping.
#[no_mangle]
pub unsafe extern "C" fn momo_rust_dequantize_q8_f32(
    input: *const i8,
    scales: *const f32,
    output: *mut f32,
    rows: usize,
    columns: usize,
) -> i32 {
    let count = match checked_product(rows, columns) {
        Some(count) => count,
        None => return INVALID_ARGUMENT,
    };
    if input.is_null() || scales.is_null() || output.is_null() {
        return INVALID_ARGUMENT;
    }
    let input = slice::from_raw_parts(input, count);
    let scales = slice::from_raw_parts(scales, rows);
    let output = slice::from_raw_parts_mut(output, count);
    for row in 0..rows {
        let scale = scales[row];
        if !scale.is_finite() || scale < f32::MIN_POSITIVE {
            return NUMERIC_ERROR;
        }
        let offset = row * columns;
        for column in 0..columns {
            output[offset + column] = f32::from(input[offset + column]) * scale;
        }
    }
    if finite_values(output) {
        OK
    } else {
        NUMERIC_ERROR
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    logit: f32,
    index: usize,
    weight: f64,
}

/// Samples a token with deterministic temperature, top-k and top-p filtering.
///
/// # Safety
///
/// `logits` must contain `count` values and `sampled_index` must be writable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn momo_rust_sample_f32(
    logits: *const f32,
    count: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    counter: u64,
    sampled_index: *mut usize,
) -> i32 {
    if logits.is_null()
        || sampled_index.is_null()
        || count == 0
        || count > isize::MAX as usize / size_of::<f32>()
        || !temperature.is_finite()
        || temperature < 0.0
        || !(0.0..=1.0).contains(&top_p)
        || top_p == 0.0
        || top_k > count
    {
        return INVALID_ARGUMENT;
    }
    let logits = slice::from_raw_parts(logits, count);
    if !finite_values(logits) {
        return NUMERIC_ERROR;
    }
    if temperature == 0.0 {
        let mut best = 0;
        for index in 1..count {
            if logits[index] > logits[best] {
                best = index;
            }
        }
        *sampled_index = best;
        return OK;
    }
    let mut candidates = Vec::new();
    if candidates.try_reserve_exact(count).is_err() {
        return OUT_OF_MEMORY;
    }
    for (index, &logit) in logits.iter().enumerate() {
        candidates.push(Candidate {
            logit: logit / temperature,
            index,
            weight: 0.0,
        });
    }
    candidates.sort_by(|left, right| {
        right
            .logit
            .partial_cmp(&left.logit)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.index.cmp(&right.index))
    });
    if top_k != 0 {
        candidates.truncate(top_k);
    }
    let maximum = f64::from(candidates[0].logit);
    let mut total = 0.0_f64;
    for candidate in &mut candidates {
        candidate.weight = (f64::from(candidate.logit) - maximum).exp();
        total += candidate.weight;
    }
    if total <= 0.0 || !total.is_finite() {
        return NUMERIC_ERROR;
    }
    let mut kept_mass = 0.0_f64;
    let mut kept = 0;
    while kept < candidates.len() {
        kept_mass += candidates[kept].weight;
        kept += 1;
        if kept_mass / total >= f64::from(top_p) {
            break;
        }
    }
    let target = uniform_sample(seed, counter) * kept_mass;
    let mut cumulative = 0.0_f64;
    for (index, candidate) in candidates.iter().take(kept).enumerate() {
        cumulative += candidate.weight;
        if target < cumulative || index + 1 == kept {
            *sampled_index = candidate.index;
            return OK;
        }
    }
    NUMERIC_ERROR
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
        return INVALID_ARGUMENT;
    }
    let input_count = batch * input_size;
    let weight_count = input_size * output_size;
    let output_count = batch * output_size;
    let residual_count = batch * residual_size;
    let residual_weight_count = residual_size * output_size;
    let input = slice::from_raw_parts(input, input_count);
    let weights = slice::from_raw_parts(weights, weight_count);
    let bias = slice::from_raw_parts(bias, output_size);
    let gate_weights = slice::from_raw_parts(gate_weights, weight_count);
    let gate_bias = slice::from_raw_parts(gate_bias, output_size);
    let residual = slice::from_raw_parts(residual, residual_count);
    let residual_weights = slice::from_raw_parts(residual_weights, residual_weight_count);
    let output = slice::from_raw_parts_mut(output, output_count);
    if !finite_values(input)
        || !finite_values(weights)
        || !finite_values(bias)
        || !finite_values(gate_weights)
        || !finite_values(gate_bias)
        || !finite_values(residual)
        || !finite_values(residual_weights)
    {
        return NUMERIC_ERROR;
    }
    let mut projection = match allocate_f32(output_count) {
        Ok(values) => values,
        Err(status) => return status,
    };
    let mut gate = match allocate_f32(output_count) {
        Ok(values) => values,
        Err(status) => return status,
    };
    let mut shortcut = match allocate_f32(output_count) {
        Ok(values) => values,
        Err(status) => return status,
    };
    if matmul(
        input,
        weights,
        &mut projection,
        batch,
        input_size,
        output_size,
    ) != OK
        || matmul(
            input,
            gate_weights,
            &mut gate,
            batch,
            input_size,
            output_size,
        ) != OK
        || matmul(
            residual,
            residual_weights,
            &mut shortcut,
            batch,
            residual_size,
            output_size,
        ) != OK
    {
        return NUMERIC_ERROR;
    }
    for row in 0..batch {
        for column in 0..output_size {
            let index = row * output_size + column;
            let projected = projection[index] + bias[column];
            let gated = gate[index] + gate_bias[column];
            if !projected.is_finite() || !gated.is_finite() {
                return NUMERIC_ERROR;
            }
            output[index] =
                activate(projected, column / group_size) * sigmoid(gated) + shortcut[index];
        }
    }
    if finite_values(output) {
        OK
    } else {
        NUMERIC_ERROR
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_multiplication_is_correct() {
        let left = [1.0, 2.0, 3.0, 4.0];
        let right = [5.0, 6.0, 7.0, 8.0];
        let mut output = [0.0; 4];
        let status = unsafe {
            momo_rust_matmul_f32(left.as_ptr(), right.as_ptr(), output.as_mut_ptr(), 2, 2, 2)
        };
        assert_eq!(status, OK);
        assert_eq!(output, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn attention_is_causal_and_sampler_is_repeatable() {
        let query = [1.0, 0.0, 1.0, 0.0];
        let key = [1.0, 0.0, 0.0, 1.0];
        let value = [3.0, 4.0, 9.0, 10.0];
        let mut output = [0.0; 4];
        let status = unsafe {
            momo_rust_causal_gqa_f32(
                query.as_ptr(),
                key.as_ptr(),
                value.as_ptr(),
                output.as_mut_ptr(),
                2,
                1,
                1,
                2,
                1.0,
            )
        };
        assert_eq!(status, OK);
        assert_eq!(&output[0..2], &[3.0, 4.0]);

        let logits = [0.0, 1.0, 2.0, 3.0];
        let mut first = 0;
        let mut second = 0;
        unsafe {
            assert_eq!(
                momo_rust_sample_f32(logits.as_ptr(), 4, 0.8, 3, 0.9, 7, 2, &mut first),
                OK
            );
            assert_eq!(
                momo_rust_sample_f32(logits.as_ptr(), 4, 0.8, 3, 0.9, 7, 2, &mut second),
                OK
            );
        }
        assert_eq!(first, second);
    }

    #[test]
    fn q8_avoids_subnormal_scales() {
        let input = [f32::from_bits(1), -f32::from_bits(1), 0.0, 0.0];
        let mut quantized = [0_i8; 4];
        let mut scales = [0.0_f32; 2];
        let status = unsafe {
            momo_rust_quantize_q8_f32(
                input.as_ptr(),
                quantized.as_mut_ptr(),
                scales.as_mut_ptr(),
                2,
                2,
            )
        };
        assert_eq!(status, OK);
        assert!(scales.iter().all(|scale| *scale >= f32::MIN_POSITIVE));
    }

    #[test]
    fn rope_preserves_high_position_bits_and_non_finite_values_fail() {
        let query = [1.0, 0.0, 1.0, 0.0];
        let key = query;
        let positions = [(1_u64 << 63) + 5, (1_u64 << 63) + 6];
        let mut query_output = [0.0; 4];
        let mut key_output = [0.0; 4];
        let status = unsafe {
            momo_rust_rope_f32(
                query.as_ptr(),
                key.as_ptr(),
                positions.as_ptr(),
                query_output.as_mut_ptr(),
                key_output.as_mut_ptr(),
                2,
                1,
                1,
                2,
                2,
                10_000.0,
            )
        };
        assert_eq!(status, OK);
        assert!((query_output[0] - query_output[2]).abs() > 1e-4);

        let invalid = [0.0, f32::INFINITY];
        let mut output = [0.0; 2];
        let status = unsafe { momo_rust_softmax_f32(invalid.as_ptr(), output.as_mut_ptr(), 1, 2) };
        assert_eq!(status, NUMERIC_ERROR);
    }
}
