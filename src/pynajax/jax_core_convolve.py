"""Vectorized one-dimensional convolution."""

from functools import partial

import jax
import jax.numpy as jnp

from .utils import _get_idxs

_convolve_vec = jax.vmap(partial(jnp.convolve, mode="same"), (1, None), 1)
_convolve_mat = jax.vmap(_convolve_vec, (None, 1), -1)


@jax.jit
def _reshape_convolve_2d_kernel(tensor, kernel):
    """
    Reshape and convolve a N-Dimensional tensor with a 2D kernel.

    Parameters
    ----------
    tensor : jax.numpy.ndarray
        The input tensor to be convolved.
    kernel : jax.numpy.ndarray
        The convolution kernel.

    Returns
    -------
    jax.numpy.ndarray:
        The convolved tensor.
    """
    shape_new = (*tensor.shape, kernel.shape[1])
    return _convolve_mat(tensor.reshape(tensor.shape[0], -1), kernel).reshape(shape_new)


@jax.jit
def _reshape_convolve_1d_kernel(tensor, kernel):
    """
    Reshape and convolve a N-Dimensional tensor with a 1D kernel.

    Parameters
    ----------
    tensor : jax.numpy.ndarray
        The input tensor to be convolved.
    kernel : jax.numpy.ndarray
        The convolution kernel.

    Returns
    -------
    jax.numpy.ndarray:
        The convolved tensor.
    """
    return _convolve_vec(tensor.reshape(tensor.shape[0], -1), kernel).reshape(
        tensor.shape
    )


@jax.jit
def _jit_tree_convolve_2d_kernel(tree, kernel):
    """
    Convolve each epoch of a tree with a 2D kernel.

    Parameters
    ----------
    tree : Any
        Tree of tensors (usually list of arrays, one array per epoch)
        to be convolved.
    kernel : jax.numpy.ndarray
        The convolution kernel.

    Returns
    -------
    jax.numpy.ndarray:
        The concatenated convolved epochs.
    """
    # Convolve each epoch
    func = partial(_reshape_convolve_2d_kernel, kernel=kernel)
    convolved_epochs = jax.tree.map(lambda x: func(x), tree)
    # Concatenate leaves on the first axis and return
    return jnp.concatenate(jax.tree.leaves(convolved_epochs), axis=0)


@jax.jit
def _jit_tree_convolve_1d_kernel(tree, kernel):
    """
    Convolve each epoch of a tree with a 1D kernel.

    Parameters
    ----------
    tree : Any
        Tree of tensors (usually list of arrays, one array per epoch) to
        be convolved.
    kernel : jax.numpy.ndarray
        The convolution kernel.

    Returns
    -------
    jax.numpy.ndarray:
        The concatenated convolved epochs.
    """
    # Convolve each epoch
    func = partial(_reshape_convolve_1d_kernel, kernel=kernel)
    convolved_epochs = jax.tree.map(lambda x: func(x), tree)
    # Concatenate leaves on the first axis and return
    return jnp.concatenate(jax.tree.leaves(convolved_epochs), axis=0)


@jax.jit
def convolve_epoch(data, kernel):
    """Convolve a single continuous temporal epoch.

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
        Pynapple timeseries object with convolved data. If kernel
        is a 1-D array, the dimensions of the input data are retained.
        If kernel is a 2-D array,  another (last) dimension is added to
        store convolution with every column of kernels.
    """
    if kernel.ndim == 1:
        data = _jit_tree_convolve_1d_kernel(data, kernel)
    else:
        data = _jit_tree_convolve_2d_kernel(data, kernel)
    return data


def convolve_intervals(time_array, data_array, starts, ends, kernel):
    """Convolve over the first dimension.

    Convolve over the first dimension, vectorizing on every dimension of data,
    with every column of kernel.

    Parameters
    ----------
    data : pynapple.Tsd, pynapple.TsdFrame, pynapple.TsdTensor
        Pynapple timeseries object with data to be convolved
    kernel : numpy.ndarray, jax.numpy.ndarray
        1-D or 2-D array with kernel(s) to be used for convolution.
        First dimension is assumed to be time.

    Returns
    -------
    : jax.ndarray
        Pynapple timeseries object with convolved data. If kernel is a
        1-D array, the dimensions of the input data are retained. If kernel
        is a 2-D array, another (last) dimension is added to store
        convolution with every column of kernels.
    """

    idx_start, idx_end = _get_idxs(time_array, starts, ends)

    tree = [data_array[start:end] for start, end in zip(idx_start, idx_end)]

    if kernel.ndim == 1:
        convolved_data = _jit_tree_convolve_1d_kernel(tree, kernel)
    else:
        convolved_data = _jit_tree_convolve_2d_kernel(tree, kernel)

    return convolved_data


def convolve(time_array, data_array, starts, ends, kernel):
    """One-dimensional convolution."""
    # Perform convolution
    if kernel.ndim == 0:
        raise IOError(
            "Provide a kernel with at least 1 dimension, current kernel has "
            "0 dimensions"
        )

    if len(starts) == 1 and len(ends) == 1:
        out = convolve_epoch(data_array, kernel)
    else:
        out = convolve_intervals(time_array, data_array, starts, ends, kernel)

    return out
