#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "momo_core.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

bool checked_product(Py_ssize_t left, Py_ssize_t right, Py_ssize_t &result) {
    if (left <= 0 || right <= 0 || left > PY_SSIZE_T_MAX / right) {
        return false;
    }
    result = left * right;
    return true;
}

bool format_matches(const char *format, const char *allowed) {
    if (format == nullptr) {
        return false;
    }
    const std::size_t length = std::strlen(format);
    return length > 0 && std::strchr(allowed, format[length - 1]) != nullptr;
}

struct Buffer {
    Py_buffer view{};
    bool acquired = false;

    ~Buffer() {
        if (acquired) {
            PyBuffer_Release(&view);
        }
    }

    bool get(PyObject *object, int dimensions, Py_ssize_t itemsize,
             const char *formats, const char *message) {
        if (PyObject_GetBuffer(object, &view,
                               PyBUF_FORMAT | PyBUF_ND | PyBUF_C_CONTIGUOUS) != 0) {
            return false;
        }
        acquired = true;
        if (view.itemsize != itemsize || view.ndim != dimensions ||
            !format_matches(view.format, formats) || view.shape == nullptr) {
            PyErr_SetString(PyExc_TypeError, message);
            return false;
        }
        Py_ssize_t elements = 1;
        for (int dimension = 0; dimension < dimensions; ++dimension) {
            if (!checked_product(elements, view.shape[dimension], elements)) {
                PyErr_SetString(PyExc_OverflowError, "native tensor shape is empty or too large");
                return false;
            }
        }
        Py_ssize_t bytes;
        if (!checked_product(elements, itemsize, bytes) || view.len != bytes) {
            PyErr_SetString(PyExc_BufferError, "native tensor buffer length does not match its shape");
            return false;
        }
        return true;
    }
};

struct FloatBuffer : Buffer {
    bool get(PyObject *object, int dimensions) {
        return Buffer::get(object, dimensions, static_cast<Py_ssize_t>(sizeof(float)),
                           "f", "expected a C-contiguous float32 tensor");
    }
    const float *data() const { return static_cast<const float *>(view.buf); }
};

struct Int8Buffer : Buffer {
    bool get(PyObject *object, int dimensions) {
        return Buffer::get(object, dimensions, static_cast<Py_ssize_t>(sizeof(std::int8_t)),
                           "b", "expected a C-contiguous int8 tensor");
    }
    const std::int8_t *data() const {
        return static_cast<const std::int8_t *>(view.buf);
    }
};

struct UInt64Buffer : Buffer {
    bool get(PyObject *object, int dimensions) {
        return Buffer::get(object, dimensions, static_cast<Py_ssize_t>(sizeof(std::uint64_t)),
                           "LQ", "expected a C-contiguous uint64 tensor");
    }
    const std::uint64_t *data() const {
        return static_cast<const std::uint64_t *>(view.buf);
    }
};

PyObject *byte_output(Py_ssize_t count, Py_ssize_t itemsize) {
    Py_ssize_t bytes;
    if (!checked_product(count, itemsize, bytes)) {
        PyErr_SetString(PyExc_OverflowError, "native output is too large");
        return nullptr;
    }
    return PyByteArray_FromStringAndSize(nullptr, bytes);
}

PyObject *float_output(Py_ssize_t count) {
    return byte_output(count, static_cast<Py_ssize_t>(sizeof(float)));
}

PyObject *raise_status(int status, const char *operation) {
    if (status == MOMO_STATUS_OUT_OF_MEMORY) {
        return PyErr_NoMemory();
    }
    PyObject *exception = PyExc_RuntimeError;
    if (status == MOMO_STATUS_INVALID_ARGUMENT) {
        exception = PyExc_ValueError;
    } else if (status == MOMO_STATUS_NUMERIC_ERROR) {
        exception = PyExc_FloatingPointError;
    } else if (status == MOMO_STATUS_OVERFLOW) {
        exception = PyExc_OverflowError;
    }
    PyErr_Format(exception, "native %s failed with status %d", operation, status);
    return nullptr;
}

PyObject *python_matmul(PyObject *, PyObject *arguments) {
    PyObject *left_object;
    PyObject *right_object;
    if (!PyArg_ParseTuple(arguments, "OO:matmul", &left_object, &right_object)) {
        return nullptr;
    }
    FloatBuffer left;
    FloatBuffer right;
    if (!left.get(left_object, 2) || !right.get(right_object, 2)) {
        return nullptr;
    }
    if (left.view.shape[1] != right.view.shape[0]) {
        PyErr_SetString(PyExc_ValueError, "matmul inner dimensions do not match");
        return nullptr;
    }
    const Py_ssize_t rows = left.view.shape[0];
    const Py_ssize_t inner = left.view.shape[1];
    const Py_ssize_t columns = right.view.shape[1];
    Py_ssize_t count;
    if (!checked_product(rows, columns, count)) {
        PyErr_SetString(PyExc_OverflowError, "matmul output shape is too large");
        return nullptr;
    }
    PyObject *result = float_output(count);
    if (result == nullptr) {
        return nullptr;
    }
    float *output = reinterpret_cast<float *>(PyByteArray_AS_STRING(result));
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = momo_matmul_f32(left.data(), right.data(), output,
                             static_cast<std::size_t>(rows),
                             static_cast<std::size_t>(inner),
                             static_cast<std::size_t>(columns));
    Py_END_ALLOW_THREADS
    if (status != MOMO_STATUS_OK) {
        Py_DECREF(result);
        return raise_status(status, "matmul");
    }
    return result;
}

PyObject *python_row_operation(PyObject *arguments, int operation) {
    PyObject *input_object;
    PyObject *weight_object = nullptr;
    double epsilon = 1e-5;
    if (operation == 0) {
        if (!PyArg_ParseTuple(arguments, "O:softmax", &input_object)) {
            return nullptr;
        }
    } else if (operation == 1) {
        if (!PyArg_ParseTuple(arguments, "O|d:layer_norm", &input_object, &epsilon)) {
            return nullptr;
        }
    } else if (!PyArg_ParseTuple(arguments, "OO|d:rms_norm", &input_object,
                                 &weight_object, &epsilon)) {
        return nullptr;
    }
    FloatBuffer input;
    FloatBuffer weight;
    if (!input.get(input_object, 2) ||
        (weight_object != nullptr && !weight.get(weight_object, 1))) {
        return nullptr;
    }
    const Py_ssize_t rows = input.view.shape[0];
    const Py_ssize_t columns = input.view.shape[1];
    if (weight_object != nullptr && weight.view.shape[0] != columns) {
        PyErr_SetString(PyExc_ValueError, "RMSNorm weight shape does not match input width");
        return nullptr;
    }
    Py_ssize_t count;
    if (!checked_product(rows, columns, count)) {
        PyErr_SetString(PyExc_OverflowError, "row operation output is too large");
        return nullptr;
    }
    PyObject *result = float_output(count);
    if (result == nullptr) {
        return nullptr;
    }
    float *output = reinterpret_cast<float *>(PyByteArray_AS_STRING(result));
    int status;
    Py_BEGIN_ALLOW_THREADS
    if (operation == 0) {
        status = momo_softmax_f32(input.data(), output,
                                  static_cast<std::size_t>(rows),
                                  static_cast<std::size_t>(columns));
    } else if (operation == 1) {
        status = momo_layer_norm_f32(input.data(), output,
                                     static_cast<std::size_t>(rows),
                                     static_cast<std::size_t>(columns),
                                     static_cast<float>(epsilon));
    } else {
        status = momo_rms_norm_f32(input.data(), weight.data(), output,
                                   static_cast<std::size_t>(rows),
                                   static_cast<std::size_t>(columns),
                                   static_cast<float>(epsilon));
    }
    Py_END_ALLOW_THREADS
    if (status != MOMO_STATUS_OK) {
        Py_DECREF(result);
        return raise_status(status, operation == 0 ? "softmax" :
                                    operation == 1 ? "layer norm" : "RMS norm");
    }
    return result;
}

PyObject *python_softmax(PyObject *, PyObject *arguments) {
    return python_row_operation(arguments, 0);
}

PyObject *python_layer_norm(PyObject *, PyObject *arguments) {
    return python_row_operation(arguments, 1);
}

PyObject *python_rms_norm(PyObject *, PyObject *arguments) {
    return python_row_operation(arguments, 2);
}

PyObject *python_rope(PyObject *, PyObject *arguments) {
    PyObject *query_object;
    PyObject *key_object;
    PyObject *positions_object;
    Py_ssize_t rotary_dimensions;
    double theta = 10000.0;
    if (!PyArg_ParseTuple(arguments, "OOOn|d:rope", &query_object, &key_object,
                          &positions_object, &rotary_dimensions, &theta)) {
        return nullptr;
    }
    FloatBuffer query;
    FloatBuffer key;
    UInt64Buffer positions;
    if (!query.get(query_object, 3) || !key.get(key_object, 3) ||
        !positions.get(positions_object, 1)) {
        return nullptr;
    }
    const Py_ssize_t tokens = query.view.shape[0];
    if (key.view.shape[0] != tokens || positions.view.shape[0] != tokens ||
        key.view.shape[2] != query.view.shape[2] || rotary_dimensions <= 0 ||
        rotary_dimensions > query.view.shape[2]) {
        PyErr_SetString(PyExc_ValueError, "RoPE tensor shapes do not match");
        return nullptr;
    }
    Py_ssize_t query_rows;
    Py_ssize_t query_count;
    Py_ssize_t key_rows;
    Py_ssize_t key_count;
    if (!checked_product(tokens, query.view.shape[1], query_rows) ||
        !checked_product(query_rows, query.view.shape[2], query_count) ||
        !checked_product(tokens, key.view.shape[1], key_rows) ||
        !checked_product(key_rows, key.view.shape[2], key_count)) {
        PyErr_SetString(PyExc_OverflowError, "RoPE output shape is too large");
        return nullptr;
    }
    PyObject *query_result = float_output(query_count);
    PyObject *key_result = float_output(key_count);
    if (query_result == nullptr || key_result == nullptr) {
        Py_XDECREF(query_result);
        Py_XDECREF(key_result);
        return nullptr;
    }
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = momo_rope_f32(
        query.data(), key.data(), positions.data(),
        reinterpret_cast<float *>(PyByteArray_AS_STRING(query_result)),
        reinterpret_cast<float *>(PyByteArray_AS_STRING(key_result)),
        static_cast<std::size_t>(tokens),
        static_cast<std::size_t>(query.view.shape[1]),
        static_cast<std::size_t>(key.view.shape[1]),
        static_cast<std::size_t>(query.view.shape[2]),
        static_cast<std::size_t>(rotary_dimensions), static_cast<float>(theta));
    Py_END_ALLOW_THREADS
    if (status != MOMO_STATUS_OK) {
        Py_DECREF(query_result);
        Py_DECREF(key_result);
        return raise_status(status, "RoPE");
    }
    PyObject *result = PyTuple_Pack(2, query_result, key_result);
    Py_DECREF(query_result);
    Py_DECREF(key_result);
    return result;
}

PyObject *python_causal_gqa(PyObject *, PyObject *arguments) {
    PyObject *query_object;
    PyObject *key_object;
    PyObject *value_object;
    double scale;
    if (!PyArg_ParseTuple(arguments, "OOOd:causal_gqa", &query_object, &key_object,
                          &value_object, &scale)) {
        return nullptr;
    }
    FloatBuffer query;
    FloatBuffer key;
    FloatBuffer value;
    if (!query.get(query_object, 3) || !key.get(key_object, 3) ||
        !value.get(value_object, 3)) {
        return nullptr;
    }
    if (query.view.shape[0] != key.view.shape[0] ||
        key.view.shape[0] != value.view.shape[0] ||
        key.view.shape[1] != value.view.shape[1] ||
        query.view.shape[2] != key.view.shape[2] ||
        key.view.shape[2] != value.view.shape[2] ||
        query.view.shape[1] % key.view.shape[1] != 0) {
        PyErr_SetString(PyExc_ValueError, "causal GQA tensor shapes do not match");
        return nullptr;
    }
    Py_ssize_t rows;
    Py_ssize_t count;
    if (!checked_product(query.view.shape[0], query.view.shape[1], rows) ||
        !checked_product(rows, query.view.shape[2], count)) {
        PyErr_SetString(PyExc_OverflowError, "causal GQA output shape is too large");
        return nullptr;
    }
    PyObject *result = float_output(count);
    if (result == nullptr) {
        return nullptr;
    }
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = momo_causal_gqa_f32(
        query.data(), key.data(), value.data(),
        reinterpret_cast<float *>(PyByteArray_AS_STRING(result)),
        static_cast<std::size_t>(query.view.shape[0]),
        static_cast<std::size_t>(query.view.shape[1]),
        static_cast<std::size_t>(key.view.shape[1]),
        static_cast<std::size_t>(query.view.shape[2]), static_cast<float>(scale));
    Py_END_ALLOW_THREADS
    if (status != MOMO_STATUS_OK) {
        Py_DECREF(result);
        return raise_status(status, "causal GQA");
    }
    return result;
}

PyObject *python_decode_attention(PyObject *, PyObject *arguments) {
    PyObject *query_object;
    PyObject *key_object;
    PyObject *value_object;
    unsigned long long position;
    double scale;
    if (!PyArg_ParseTuple(arguments, "OOOKd:decode_attention", &query_object,
                          &key_object, &value_object, &position, &scale)) {
        return nullptr;
    }
    FloatBuffer query;
    FloatBuffer key;
    FloatBuffer value;
    if (!query.get(query_object, 2) || !key.get(key_object, 3) ||
        !value.get(value_object, 3)) {
        return nullptr;
    }
    if (key.view.shape[0] != value.view.shape[0] ||
        key.view.shape[1] != value.view.shape[1] ||
        query.view.shape[1] != key.view.shape[2] ||
        key.view.shape[2] != value.view.shape[2] ||
        query.view.shape[0] % key.view.shape[1] != 0) {
        PyErr_SetString(PyExc_ValueError, "decode attention tensor shapes do not match");
        return nullptr;
    }
    Py_ssize_t count;
    if (!checked_product(query.view.shape[0], query.view.shape[1], count)) {
        PyErr_SetString(PyExc_OverflowError, "decode attention output shape is too large");
        return nullptr;
    }
    PyObject *result = float_output(count);
    if (result == nullptr) {
        return nullptr;
    }
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = momo_decode_attention_f32(
        query.data(), key.data(), value.data(),
        reinterpret_cast<float *>(PyByteArray_AS_STRING(result)),
        static_cast<std::size_t>(key.view.shape[0]),
        static_cast<std::size_t>(query.view.shape[0]),
        static_cast<std::size_t>(key.view.shape[1]),
        static_cast<std::size_t>(query.view.shape[1]),
        static_cast<std::uint64_t>(position), static_cast<float>(scale));
    Py_END_ALLOW_THREADS
    if (status != MOMO_STATUS_OK) {
        Py_DECREF(result);
        return raise_status(status, "decode attention");
    }
    return result;
}

PyObject *python_quantize_q8(PyObject *, PyObject *arguments) {
    PyObject *input_object;
    if (!PyArg_ParseTuple(arguments, "O:quantize_q8", &input_object)) {
        return nullptr;
    }
    FloatBuffer input;
    if (!input.get(input_object, 2)) {
        return nullptr;
    }
    Py_ssize_t count;
    if (!checked_product(input.view.shape[0], input.view.shape[1], count)) {
        PyErr_SetString(PyExc_OverflowError, "Q8 output shape is too large");
        return nullptr;
    }
    PyObject *quantized = byte_output(count, 1);
    PyObject *scales = float_output(input.view.shape[0]);
    if (quantized == nullptr || scales == nullptr) {
        Py_XDECREF(quantized);
        Py_XDECREF(scales);
        return nullptr;
    }
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = momo_quantize_q8_f32(
        input.data(),
        reinterpret_cast<std::int8_t *>(PyByteArray_AS_STRING(quantized)),
        reinterpret_cast<float *>(PyByteArray_AS_STRING(scales)),
        static_cast<std::size_t>(input.view.shape[0]),
        static_cast<std::size_t>(input.view.shape[1]));
    Py_END_ALLOW_THREADS
    if (status != MOMO_STATUS_OK) {
        Py_DECREF(quantized);
        Py_DECREF(scales);
        return raise_status(status, "Q8 quantization");
    }
    PyObject *result = PyTuple_Pack(2, quantized, scales);
    Py_DECREF(quantized);
    Py_DECREF(scales);
    return result;
}

PyObject *python_dequantize_q8(PyObject *, PyObject *arguments) {
    PyObject *input_object;
    PyObject *scales_object;
    if (!PyArg_ParseTuple(arguments, "OO:dequantize_q8", &input_object,
                          &scales_object)) {
        return nullptr;
    }
    Int8Buffer input;
    FloatBuffer scales;
    if (!input.get(input_object, 2) || !scales.get(scales_object, 1)) {
        return nullptr;
    }
    if (scales.view.shape[0] != input.view.shape[0]) {
        PyErr_SetString(PyExc_ValueError, "Q8 scale shape does not match input rows");
        return nullptr;
    }
    Py_ssize_t count;
    if (!checked_product(input.view.shape[0], input.view.shape[1], count)) {
        PyErr_SetString(PyExc_OverflowError, "Q8 output shape is too large");
        return nullptr;
    }
    PyObject *result = float_output(count);
    if (result == nullptr) {
        return nullptr;
    }
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = momo_dequantize_q8_f32(
        input.data(), scales.data(),
        reinterpret_cast<float *>(PyByteArray_AS_STRING(result)),
        static_cast<std::size_t>(input.view.shape[0]),
        static_cast<std::size_t>(input.view.shape[1]));
    Py_END_ALLOW_THREADS
    if (status != MOMO_STATUS_OK) {
        Py_DECREF(result);
        return raise_status(status, "Q8 dequantization");
    }
    return result;
}

PyObject *python_sample(PyObject *, PyObject *arguments) {
    PyObject *logits_object;
    double temperature;
    Py_ssize_t top_k;
    double top_p;
    unsigned long long seed;
    unsigned long long counter;
    if (!PyArg_ParseTuple(arguments, "OdndKK:sample", &logits_object,
                          &temperature, &top_k, &top_p, &seed, &counter)) {
        return nullptr;
    }
    FloatBuffer logits;
    if (!logits.get(logits_object, 1)) {
        return nullptr;
    }
    if (top_k < 0) {
        PyErr_SetString(PyExc_ValueError, "top_k cannot be negative");
        return nullptr;
    }
    std::size_t sampled = 0;
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = momo_sample_f32(logits.data(),
                             static_cast<std::size_t>(logits.view.shape[0]),
                             static_cast<float>(temperature),
                             static_cast<std::size_t>(top_k),
                             static_cast<float>(top_p),
                             static_cast<std::uint64_t>(seed),
                             static_cast<std::uint64_t>(counter), &sampled);
    Py_END_ALLOW_THREADS
    if (status != MOMO_STATUS_OK) {
        return raise_status(status, "sampling");
    }
    return PyLong_FromSize_t(sampled);
}

PyObject *python_neuron_group(PyObject *, PyObject *arguments) {
    PyObject *input_object;
    PyObject *weights_object;
    PyObject *bias_object;
    PyObject *gate_weights_object;
    PyObject *gate_bias_object;
    PyObject *residual_object;
    PyObject *residual_weights_object;
    Py_ssize_t group_size;
    if (!PyArg_ParseTuple(arguments, "OOOOOOOn:neuron_group", &input_object,
                          &weights_object, &bias_object, &gate_weights_object,
                          &gate_bias_object, &residual_object,
                          &residual_weights_object, &group_size)) {
        return nullptr;
    }
    FloatBuffer input;
    FloatBuffer weights;
    FloatBuffer bias;
    FloatBuffer gate_weights;
    FloatBuffer gate_bias;
    FloatBuffer residual;
    FloatBuffer residual_weights;
    if (!input.get(input_object, 2) || !weights.get(weights_object, 2) ||
        !bias.get(bias_object, 1) || !gate_weights.get(gate_weights_object, 2) ||
        !gate_bias.get(gate_bias_object, 1) || !residual.get(residual_object, 2) ||
        !residual_weights.get(residual_weights_object, 2)) {
        return nullptr;
    }
    const Py_ssize_t batch = input.view.shape[0];
    const Py_ssize_t input_size = input.view.shape[1];
    const Py_ssize_t output_size = weights.view.shape[1];
    const Py_ssize_t residual_size = residual.view.shape[1];
    if (group_size <= 0 || weights.view.shape[0] != input_size ||
        bias.view.shape[0] != output_size ||
        gate_weights.view.shape[0] != input_size ||
        gate_weights.view.shape[1] != output_size ||
        gate_bias.view.shape[0] != output_size || residual.view.shape[0] != batch ||
        residual_weights.view.shape[0] != residual_size ||
        residual_weights.view.shape[1] != output_size) {
        PyErr_SetString(PyExc_ValueError, "neuron group tensor shapes do not match");
        return nullptr;
    }
    Py_ssize_t count;
    if (!checked_product(batch, output_size, count)) {
        PyErr_SetString(PyExc_OverflowError, "neuron group output shape is too large");
        return nullptr;
    }
    PyObject *result = float_output(count);
    if (result == nullptr) {
        return nullptr;
    }
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = momo_neuron_group_f32(
        input.data(), weights.data(), bias.data(), gate_weights.data(),
        gate_bias.data(), residual.data(), residual_weights.data(),
        reinterpret_cast<float *>(PyByteArray_AS_STRING(result)),
        static_cast<std::size_t>(batch), static_cast<std::size_t>(input_size),
        static_cast<std::size_t>(output_size),
        static_cast<std::size_t>(residual_size),
        static_cast<std::size_t>(group_size));
    Py_END_ALLOW_THREADS
    if (status != MOMO_STATUS_OK) {
        Py_DECREF(result);
        return raise_status(status, "neuron group");
    }
    return result;
}

PyObject *python_backend_info(PyObject *, PyObject *) {
    return Py_BuildValue(
        "{s:s,s:i,s:s,s:(ssssssssss)}", "name", momo_cpp_backend_name(), "abi",
        MOMO_CORE_ABI_VERSION, "precision", "float32", "kernels", "matmul",
        "softmax", "layer_norm", "rms_norm", "rope", "causal_gqa",
        "decode_attention", "q8", "sampler", "neuron_group");
}

PyMethodDef methods[] = {
    {"matmul", python_matmul, METH_VARARGS, "Multiply float32 matrices."},
    {"softmax", python_softmax, METH_VARARGS, "Apply stable row softmax."},
    {"layer_norm", python_layer_norm, METH_VARARGS, "Apply row layer normalization."},
    {"rms_norm", python_rms_norm, METH_VARARGS, "Apply weighted RMS normalization."},
    {"rope", python_rope, METH_VARARGS, "Apply rotary position embeddings."},
    {"causal_gqa", python_causal_gqa, METH_VARARGS, "Run online causal grouped-query attention."},
    {"decode_attention", python_decode_attention, METH_VARARGS, "Run one-token cache attention."},
    {"quantize_q8", python_quantize_q8, METH_VARARGS, "Quantize rows to symmetric int8."},
    {"dequantize_q8", python_dequantize_q8, METH_VARARGS, "Dequantize symmetric int8 rows."},
    {"sample", python_sample, METH_VARARGS, "Sample deterministically from logits."},
    {"neuron_group", python_neuron_group, METH_VARARGS, "Run a fused mixed-activation neuron group."},
    {"backend_info", python_backend_info, METH_NOARGS, "Return native core metadata."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Momo-LM C/C++ tensor and inference kernels.",
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__native(void) {
    return PyModule_Create(&module);
}
