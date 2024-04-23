import numpy as np
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

