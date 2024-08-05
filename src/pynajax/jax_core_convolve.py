"""Vectorized one-dimensional convolution."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .utils import _get_idxs, _get_slicing

_convolve_vec = jax.vmap(partial(jnp.convolve, mode="full"), (1, None), 1)
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
    out = _convolve_mat(tensor.reshape(tensor.shape[0], -1), kernel)
    return out.reshape(
        (tensor.shape[0] + kernel.shape[0] - 1,) + tensor.shape[1:] + (kernel.shape[1],)
    )


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
    out = _convolve_vec(tensor.reshape(tensor.shape[0], -1), kernel)
    return out.reshape((tensor.shape[0] + kernel.shape[0] - 1,) + tensor.shape[1:])


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


def _get_trim_idx(t, k, trim="both"):
    if trim == "both":
        cut = ((k - 1) // 2, t + k - 1 - ((k - 1) // 2) - (1 - k % 2))
    elif trim == "left":
        cut = (k - 1, t + k - 1)
    elif trim == "right":
        cut = (0, t)
    return cut


def convolve_intervals(time_array, data_array, starts, ends, kernel, trim="both"):
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
    extra = _get_trim_idx(0, kernel.shape[0], trim)
    extra = (extra[0], extra[1] + 1)

    n = len(starts)
    idx_start_shift = (
        idx_start + np.arange(1, n + 1) * extra[0] + np.arange(0, n) * extra[1]
    )
    idx_end_shift = (
        idx_end + np.arange(1, n + 1) * extra[0] + np.arange(0, n) * extra[1]
    )

    idx = _get_slicing(idx_start_shift, idx_end_shift)

    tree = [data_array[start:end] for start, end in zip(idx_start, idx_end)]

    if kernel.ndim == 1:
        convolved_data = _jit_tree_convolve_1d_kernel(tree, kernel)
    else:
        convolved_data = _jit_tree_convolve_2d_kernel(tree, kernel)

    return convolved_data[idx]


def convolve(time_array, data_array, starts, ends, kernel, trim="both"):
    """One-dimensional convolution."""

    if not isinstance(data_array, jnp.ndarray):
        data_array = jnp.asarray(data_array)

    # Perform convolution
    if kernel.ndim == 0:
        raise IOError(
            "Provide a kernel with at least 1 dimension, current kernel has 0 dimensions"
        )

    if len(starts) == 1 and len(ends) == 1:
        cut = _get_trim_idx(data_array.shape[0], kernel.shape[0], trim)
        out = convolve_epoch(data_array, kernel)[cut[0] : cut[1]]
    else:
        out = convolve_intervals(time_array, data_array, starts, ends, kernel, trim)

    return out
