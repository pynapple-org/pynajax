from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap

_convolve_vec = jax.vmap(partial(jnp.convolve, mode="same"), (1, None), 1)
_convolve_mat = jax.vmap(_convolve_vec, (None, 1), -1)


def construct_nap(time, data, time_support, columns):
    """

    Parameters
    ----------
    time
    data
    time_support
    columns

    Returns
    -------

    """
    if data.ndim == 1:
        data = nap.Tsd(t=time, d=data, time_support=time_support)
    elif data.ndim == 2:
        data = nap.TsdFrame(t=time, d=data, columns=columns, time_support=time_support)
    else:
        data = nap.TsdTensor(t=time, d=data, time_support=time_support)
    return data


def convolve_epoch(data, kernel):
    """Convolves a single continuous temporal epoch

    Parameters
    ----------
    data : pynapple.Tsd, pynapple.TsdFrame, pynapple.TsdTensor
        Pynapple timeseries object with data to be convolved
    kernel : numpy.ndarray, jax.numpy.ndarray
        1-D or 2-D array with kernel(s) to be used for convolution.
        First dimension is assumed to be time.

    Returns
    -------
    pynapple.Tsd, pynapple.TsdFrame, pynapple.TsdTensor
        Pynapple timeseries object with convolved data. If kernel is a 1-D array,
        the dimensions of the input data are retained. If kernel is a 2-D array,
        another (last) dimension is added to store convolution with every column of kernels.
    """

    # Store extra information of the pynapple object to add back later
    orig_shape = data.shape
    time = data.t
    time_support = data.time_support
    columns = None
    if data.ndim == 2:
        columns = data.columns

    # Flatten all dimensions except for the first (time) dimension
    data = jnp.reshape(data.d, (data.shape[0], -1))

    # Perform convolution
    if kernel.ndim == 0:
        raise IOError(
            "Provide a kernel with at least 1 dimension, current kernel has 0 dimensions"
        )
    if kernel.ndim == 1:
        data = _convolve_vec(data, kernel).reshape(orig_shape)
    else:
        data = _convolve_mat(data, kernel).reshape((*orig_shape, kernel.shape[1]))

    # Recreate pynapple object
    data = construct_nap(time, data, time_support, columns)
    return data


def convolve_intervals(data, kernel):
    """Convolves every dimension of data, except for the first which
    is assumed to be time, with every column of kernel.

    Parameters
    ----------
    data : pynapple.Tsd, pynapple.TsdFrame, pynapple.TsdTensor
        Pynapple timeseries object with data to be convolved
    kernel : numpy.ndarray, jax.numpy.ndarray
        1-D or 2-D array with kernel(s) to be used for convolution.
        First dimension is assumed to be time.

    Returns
    -------
    pynapple.Tsd, pynapple.TsdFrame, pynapple.TsdTensor
        Pynapple timeseries object with convolved data. If kernel is a 1-D array,
        the dimensions of the input data are retained. If kernel is a 2-D array,
        another (last) dimension is added to store convolution with every column of kernels.
    """

    # Create a tree of pynapple timeseries objects for each epoch
    tree = [data.get(start, end) for start, end in data.time_support.values]

    # Convolve each epoch
    func = partial(convolve_epoch, kernel=kernel)
    convolved_epochs = jax.tree_map(lambda x: func(x).d, tree)

    # Concatenate the convolved epochs
    convolved_data = jnp.concatenate(convolved_epochs, axis=0)

    # Reconstruct the timeseries object
    columns = None
    if kernel.ndim == 1 and hasattr(data, "columns"):
        columns = data.columns

    return construct_nap(data.t, convolved_data, data.time_support, columns)


def convolve(data, kernel):
    if len(data.time_support) > 1:
        return convolve_epoch(data, kernel)
    return convolve_intervals(data, kernel)