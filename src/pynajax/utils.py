import numpy as np
import jax.numpy as jnp
from numba import jit


@jit(nopython=True)
def _get_idxs(time_array, starts, ends):
    """
    Find indices in a sorted time array where each start and end time should be inserted.

    This function efficiently computes the insertion points for each start and end time
    into a sorted time array. The indices for start times are found such that each start time
    is inserted before the first element greater than the start time. The indices for end times
    are found such that each end time is inserted before the first element strictly greater than
    the end time.

    Parameters
    ----------
    time_array : array_like
        The sorted array of timestamps.
    starts : array_like
        An array of start times for which indices are required in `time_array`.
    ends : array_like
        An array of end times for which indices are required in `time_array`.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing two arrays: the indices in `time_array` where the `starts`
        should be inserted, and the indices where the `ends` should be inserted.

    Examples
    --------
    >>> time_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> starts = np.array([2, 3, 5])
    >>> ends = np.array([2.9, 4.9, 7])
    >>> idx_start, idx_end = _get_idxs(time_array, starts, ends)
    >>> print(idx_start, idx_end)  # prints [1 2 4] [2 4 7]
    """
    idx_start = np.searchsorted(time_array, starts)
    idx_end = np.searchsorted(time_array, ends, side="right")
    return idx_start, idx_end

def _fill_forward(time_target_array, time_array, data_array, starts, ends, out_of_range=np.nan):
    """
    Fill a time series forward in time with data. Useful for event trigger average
    if the data have a lower sampling rate. This function assumes that the data have
    been restricted beforehand.
    
    Parameters
    ----------
    time_target_array : ArrayLike
        The time to match
    time_array : ArrayLike
        The time of the data to extent
    data_array : ArrayLike
        The data to extent
    starts : ArrayLike
        
    ends : ArrayLike
        
    out_of_range : Number, optional
        How to fill the gap
    
    Returns
    -------
    : ArrayLike
        The data time series filled forward.
        
    """
    filled_d = np.full(
        (time_target_array.shape[0], *data_array.shape[1:]), out_of_range, dtype=data_array.dtype
        )
    fill_idx = 0

    idx_start_target, idx_end_target = _get_idxs(time_target_array, starts, ends)
    idx_start, idx_end = _get_idxs(time_array, starts, ends)

    for i in range(len(idx_start)):
        d = data_array[idx_start[i]:idx_end[i]]
        t = time_array[idx_start[i]:idx_end[i]]
        ts = time_target_array[idx_start_target[i]:idx_end_target[i]]

        idxs = np.searchsorted(t, ts, side="right") - 1
        filled_d[fill_idx : fill_idx + len(ts)][idxs >= 0] = d[idxs[idxs >= 0]]
        # filled_d = filled_d.at[fill_idx : fill_idx + len(ts)][idxs >= 0].set(d[idxs[idxs >= 0]])
        fill_idx += len(ts)

    return filled_d

def _get_shifted_indices(idx_start, idx_end, window):
    """
    Compute shifted indices for given start and end indices based on a specified window size.

    Parameters
    ----------
    idx_start : ArrayLike
        Array of start indices.
    idx_end : ArrayLike
        Array of end indices.
    windows : ArrayLike
        

    Returns
    -------
    : tuple of ArrayLike
        A tuple containing two arrays: shifted start indices and shifted end indices.
    """
    cum_delta = np.concatenate([[0], np.cumsum(idx_end - idx_start)])
    idx_start_shift = window * np.arange(idx_start.shape[0]) + cum_delta[:-1]
    idx_end_shift = window * np.arange(idx_end.shape[0]) + cum_delta[1:]
    return idx_start_shift, idx_end_shift

@jit(nopython=True)
def _get_slicing(idx_start, idx_end):
    """
    Generate an array of indices for slicing, based on provided start and end indices.

    Parameters
    ----------
    idx_start : ArrayLike
        Array of start indices.
    idx_end : ArrayLike
        Array of end indices.

    Returns
    -------
    : ArrayLike
        An array of indices constructed from the start and end indices.
    """
    iend = np.sum(idx_end - idx_start)
    ix = np.zeros(iend, dtype=np.int32)
    cnt = 0
    for st, en in zip(idx_start, idx_end):
        for k in range(st, en):
            ix[cnt] = k
            cnt += 1
    return ix