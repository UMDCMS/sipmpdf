"""

kernel.py

Helper class to ensure that numpy and tensorflow has feature parity. As numpy is
easier more feature rich and easier to debug.

"""

# Place here to suppress warning message
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

import numpy
import scipy.special
import tensorflow
import zfit

# Additional names for numpy methods to map special function names. These should
# be simple 1-liner functions
__numpy_attr_map__ = {
    "loggamma": scipy.special.loggamma,
    "erf": scipy.special.erf,
    "reduce_max": numpy.max,
    "reduce_min": numpy.min,
    "toint32": lambda x: numpy.array(x).astype(numpy.int32),
    "tofloat64": lambda x: numpy.array(x).astype(numpy.float64),
    "make_array": lambda x: numpy.array(x),
}

_ = [setattr(numpy, k, v) for k, v in __numpy_attr_map__.items()]

# Additional names for tensorflow methods to special methods. These shold be
# simple 1-linear implementations.
__tensorflow_attr_map__ = {
    "loggamma": tensorflow.math.lgamma,
    "pi": tensorflow.constant(numpy.pi, tensorflow.float64),
    "toint32": lambda x: tensorflow.cast(x, tensorflow.int32),
    "tofloat64": lambda x: tensorflow.cast(x, tensorflow.float64),
    "ndim": tensorflow.rank,
    "make_array": lambda x: tensorflow.convert_to_tensor(x),
    "sum": tensorflow.math.reduce_sum,
    "power": tensorflow.math.pow,
}
_ = [setattr(tensorflow, k, v) for k, v in __tensorflow_attr_map__.items()]


class kernel_switch:
    def __init__(self, *args, **kwargs):
        args_list = [*args, *kwargs.values()]
        has_np = any([isinstance(x, numpy.ndarray) for x in args_list])
        has_tf = any(
            [
                (
                    isinstance(x, tensorflow.Tensor)
                    or isinstance(x, zfit.core.parameter.Parameter)
                )
                for x in args_list
            ]
        )
        if has_np and has_tf:
            warnings.warn(
                """
                Running both numpy and tensorflow arrays, we will use numpy.
                This may have performance penalties if you you are loading in
                large arrays!
                """,
                UserWarning,
            )

        self._default_lib = tensorflow if (has_tf and not has_np) else numpy

    def __getattr__(self, attr_name):
        """
        Attempting to get an attribute by default it will search according to the
        default library
        """
        if hasattr(self._default_lib, attr_name):
            return getattr(self._default_lib, attr_name)

        # Additional searches for tensorflow
        if self._default_lib is tensorflow:
            if hasattr(self._default_lib.math, attr_name):
                return getattr(self._default_lib.math, attr_name)
            if hasattr(self._default_lib.experimental.numpy, attr_name):
                warnings.warn(
                    f"Using tensorflow.experimental.numpy.{attr_name}. Make sure the functionality has been implemented!"
                )
                return getattr(self._default_lib.experimental.numpy, attr_name)

        # Checking local implementations

        # Common implementation that has same syntax between numpy and
        # tensorflow and should be implemented as non-static method, as it
        # requires the _default_lib object to switch between kernels
        if self._default_lib is numpy:
            if hasattr(kernel_switch, "_np_" + attr_name):
                return getattr(kernel_switch, "_np_" + attr_name)
        if self._default_lib is tensorflow:
            if hasattr(kernel_switch, "_tf_" + attr_name):
                return getattr(kernel_switch, "_tf_" + attr_name)

        if hasattr(self, "_common_" + attr_name):
            return getattr(self, "_common_" + attr_name)
        else:
            raise AttributeError(
                f"Method {attr_name} not found in module {self._default_lib.__name__}! And has not been implemented by the helper class"
            )

    """
    Additional functions that are more simple 1-liners
    """

    def _common_repeat_axis0(self, array, n_repeat):
        """
        Extending the array at the outer-most dimension by repeating the content a
        specific number of times
        """
        return self._default_lib.repeat(
            array[self._default_lib.newaxis, ...], n_repeat, axis=0
        )

    @staticmethod
    def _np_local_index(arr, axis=0):
        return numpy.indices(numpy.shape(arr))[axis]

    _local_index_einsuminfo_ = {
        0: "",
        1: "a",
        2: "ab",
        3: "abc",
        4: "abcd",
        5: "abcde",
    }

    @staticmethod
    @tensorflow.function
    def _tf_local_index(arr, axis=0):
        if abs(axis) > 5:
            raise NotImplementedError(
                "No Implementations nested dimension great than 5"
            )

        # Getting the initial index
        index = tensorflow.range(0, tensorflow.shape(arr)[axis], dtype=arr.dtype)

        if axis >= 0:
            pre = kernel_switch._local_index_einsuminfo_[axis]
            return tensorflow.einsum(
                f"i,{pre}i...->{pre}i...", index, tensorflow.ones_like(arr)  #
            )
        else:
            post = kernel_switch._local_index_einsuminfo_[-(axis + 1)]
            return tensorflow.einsum(
                f"i,...i{post}->...i{post}", index, tensorflow.ones_like(arr)  #
            )
        raise NotImplementedError("Does not know hot to handled rank6 tensors!")


def expand_shape(np_tf_func):
    """
    Decorator to expanding all arrays to the one with the most complicate shape,
    this ensures all in the calculation will have compatible dimensions. Notice
    that this method is potentially memory intensive, so do not invoke unless it
    is necessary.
    """

    def _expand_shape_func(*args, **kwargs):
        kern = kernel_switch(*args, **kwargs)
        # Assuming we will be working with no-more than rank 5 tensors #
        # TODO: generalized arbitrary dimensions.
        args_shape = kern.make_array(
            [kern.pad(kern.shape(x), [(0, 5)])[:5] for x in [*args, *kwargs.values()]]
        )
        args_rank = kern.make_array([kern.ndim(x) for x in [*args, *kwargs.values()]])
        max_dim_index = kern.argmax(args_rank)
        max_dim_shape = args_shape[max_dim_index]
        max_dim_arr = kern.tofloat64(
            kern.ones(shape=kern.toint32(max_dim_shape[max_dim_shape != 0]))
        )
        args = [x * max_dim_arr for x in args]
        kwargs = {k: v * max_dim_arr for k, v in kwargs.items()}
        return np_tf_func(*args, **kwargs)

    return _expand_shape_func
