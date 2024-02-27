from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap


_convolve_vec = jax.vmap(partial(jnp.convolve, mode="same"), (1, None), 1)
_convolve_mat = jax.vmap(_convolve_vec, (None, 1), -1)


def convolve(data, kernel):

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
        raise IOError("Provide a kernel with at least 1 dimension, current kernel has 0 dimensions")
    elif kernel.ndim == 1:
        data = _convolve_vec(data, kernel).reshape(orig_shape)
    else:
        data = _convolve_mat(data, kernel).reshape((*orig_shape, kernel.shape[1]))


    # Recreate pynapple object
    data = np.asarray(data)
    if data.ndim == 1:
        data = nap.Tsd(t=time, d=data, time_support=time_support)
    elif data.ndim == 2:
        data = nap.TsdFrame(t=time, d=data, columns=columns, time_support=time_support)
    else:
        data = nap.TsdTensor(t=time, d=data, time_support=time_support)

    return data

