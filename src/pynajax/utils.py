import jax
import jax.numpy as jnp
import numpy as np
from numba import jit


@jit(nopython=True)
def _get_idxs(time_array, starts, ends):
    idx_start = np.searchsorted(time_array, starts)
    idx_end = np.searchsorted(time_array, ends, side="right")
    return idx_start, idx_end
