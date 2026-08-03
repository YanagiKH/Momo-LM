#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "momo_core.h"

#include <cstddef>
#include <cstring>

namespace {

struct FloatBuffer {
    Py_buffer view{};
    bool acquired = false;

    ~FloatBuffer() {
        if (acquired) {
            PyBuffer_Release(&view);
        }
    }

    bool get(PyObject *object, int dimensions) {
        if (PyObject_GetBuffer(object, &view, PyBUF_FORMAT | PyBUF_ND | PyBUF_C_CONTIGUOUS) != 0) {
            return false;
        }
        acquired = true;
        if (view.itemsize != static_cast<Py_ssize_t>(sizeof(float)) || view.ndim != dimensions ||
            view.format == nullptr || (std::strcmp(view.format, "f") != 0 && std::strcmp(view.format, "=f") != 0)) {
            PyErr_SetString(PyExc_TypeError, "expected a C-contiguous float32 buffer with the required dimensions");
            return false;
        }
        return true;
    }

    const float *data() const { return static_cast<const float *>(view.buf); }
};

PyObject *float_output(Py_ssize_t count) {
    return PyByteArray_FromStringAndSize(nullptr, count * static_cast<Py_ssize_t>(sizeof(float)));
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
    PyObject *result = float_output(rows * columns);
    if (result == nullptr) {
        return nullptr;
    }
    float *output = reinterpret_cast<float *>(PyByteArray_AS_STRING(result));
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = momo_matmul_f32(left.data(), right.data(), output,
                             static_cast<size_t>(rows), static_cast<size_t>(inner),
                             static_cast<size_t>(columns));
    Py_END_ALLOW_THREADS
    if (status != 0) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_RuntimeError, "native matmul failed");
        return nullptr;
    }
    return result;
}

PyObject *python_row_operation(PyObject *arguments, bool layer_norm) {
    PyObject *input_object;
    double epsilon = 1e-5;
    if (layer_norm) {
        if (!PyArg_ParseTuple(arguments, "O|d:layer_norm", &input_object, &epsilon)) {
            return nullptr;
        }
    } else if (!PyArg_ParseTuple(arguments, "O:softmax", &input_object)) {
        return nullptr;
    }
    FloatBuffer input;
    if (!input.get(input_object, 2)) {
        return nullptr;
    }
    const Py_ssize_t rows = input.view.shape[0];
    const Py_ssize_t columns = input.view.shape[1];
    PyObject *result = float_output(rows * columns);
    if (result == nullptr) {
        return nullptr;
    }
    float *output = reinterpret_cast<float *>(PyByteArray_AS_STRING(result));
    int status;
    Py_BEGIN_ALLOW_THREADS
    if (layer_norm) {
        status = momo_layer_norm_f32(input.data(), output, static_cast<size_t>(rows),
                                     static_cast<size_t>(columns), static_cast<float>(epsilon));
    } else {
        status = momo_softmax_f32(input.data(), output, static_cast<size_t>(rows),
                                  static_cast<size_t>(columns));
    }
    Py_END_ALLOW_THREADS
    if (status != 0) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_RuntimeError, "native row operation failed");
        return nullptr;
    }
    return result;
}

PyObject *python_softmax(PyObject *, PyObject *arguments) {
    return python_row_operation(arguments, false);
}

PyObject *python_layer_norm(PyObject *, PyObject *arguments) {
    return python_row_operation(arguments, true);
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
    if (!PyArg_ParseTuple(arguments, "OOOOOOOn:neuron_group", &input_object, &weights_object,
                          &bias_object, &gate_weights_object, &gate_bias_object,
                          &residual_object, &residual_weights_object, &group_size)) {
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
    if (group_size <= 0 || weights.view.shape[0] != input_size || bias.view.shape[0] != output_size ||
        gate_weights.view.shape[0] != input_size || gate_weights.view.shape[1] != output_size ||
        gate_bias.view.shape[0] != output_size || residual.view.shape[0] != batch ||
        residual_weights.view.shape[0] != residual_size || residual_weights.view.shape[1] != output_size) {
        PyErr_SetString(PyExc_ValueError, "neuron group tensor shapes do not match");
        return nullptr;
    }
    PyObject *result = float_output(batch * output_size);
    if (result == nullptr) {
        return nullptr;
    }
    float *output = reinterpret_cast<float *>(PyByteArray_AS_STRING(result));
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = momo_neuron_group_f32(
        input.data(), weights.data(), bias.data(), gate_weights.data(), gate_bias.data(),
        residual.data(), residual_weights.data(), output, static_cast<size_t>(batch),
        static_cast<size_t>(input_size), static_cast<size_t>(output_size),
        static_cast<size_t>(residual_size), static_cast<size_t>(group_size));
    Py_END_ALLOW_THREADS
    if (status != 0) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_RuntimeError, "native neuron group failed");
        return nullptr;
    }
    return result;
}

PyObject *python_backend_info(PyObject *, PyObject *) {
    return Py_BuildValue("{s:s,s:i,s:s}", "name", momo_cpp_backend_name(), "abi",
                         MOMO_CORE_ABI_VERSION, "precision", "float32");
}

PyMethodDef methods[] = {
    {"matmul", python_matmul, METH_VARARGS, "Multiply two C-contiguous float32 matrices."},
    {"softmax", python_softmax, METH_VARARGS, "Apply stable row-wise softmax."},
    {"layer_norm", python_layer_norm, METH_VARARGS, "Apply row-wise layer normalization."},
    {"neuron_group", python_neuron_group, METH_VARARGS, "Run the fused mixed-activation neuron group."},
    {"backend_info", python_backend_info, METH_NOARGS, "Return native core metadata."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Momo-LM C/C++ tensor and inference kernels.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__native(void) {
    return PyModule_Create(&module);
}
