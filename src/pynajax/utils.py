import jax
import jax.numpy as jnp
from numba import jit
import numpy as np


@jit(nopython=True)
def _get_idxs(time_array, starts, ends):
    idx_start = jnp.searchsorted(time_array, starts)
    idx_end = jnp.searchsorted(time_array, ends, side="right")
    return idx_start, idx_end


def _get_numpy_idxs(time_array, starts, ends):
    idx_start = np.searchsorted(time_array, starts)
    idx_end = np.searchsorted(time_array, ends, side="right")
    return idx_start, idx_end