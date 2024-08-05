import jax.numpy as jnp

from .utils import _get_idxs, _get_slicing


def threshold(time_array, data_array, starts, ends, thr, method):
    """Threshold function for pynajax

    Parameters
    ----------
    time_array : ArrayLike

    data_array : ArrayLike

    starts : ArrayLike

    ends : ArrayLike

    thr : Number

    method : string


    Returns
    -------
    tuple of ArrayLike
        Description
    """
    if not isinstance(data_array, jnp.ndarray):
        data_array = jnp.asarray(data_array)

    idx_start, idx_end = _get_idxs(time_array, starts, ends)
    idx_slicing = _get_slicing(idx_start, idx_end)

    data_array = data_array[idx_slicing]
    time_array = time_array[idx_slicing]

    if method == "above":
        ix = data_array > thr
    elif method == "below":
        ix = data_array < thr
    elif method == "aboveequal":
        ix = data_array >= thr
    elif method == "belowequal":
        ix = data_array <= thr

    ix2 = jnp.diff(ix * 1)

    new_starts = (
        time_array[1:][ix2 == 1]
        - (time_array[1:][ix2 == 1] - time_array[0:-1][ix2 == 1]) / 2
    )
    new_ends = (
        time_array[0:-1][ix2 == -1]
        + (time_array[1:][ix2 == -1] - time_array[0:-1][ix2 == -1]) / 2
    )

    if ix[0]:  # First element to keep as start
        new_starts = jnp.hstack((jnp.array([time_array[0]]), new_starts))
    if ix[-1]:  # last element to keep as end
        new_ends = jnp.hstack((new_ends, jnp.array([time_array[-1]])))

    return time_array[ix], data_array[ix], new_starts, new_ends
