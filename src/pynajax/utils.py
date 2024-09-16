import warnings
from numbers import Number

import jax
import jax.numpy as jnp
import numpy as np
from numba import jit


def is_array_like(obj):
    """
    Check if an object is array-like.

    This function determines if an object has array-like properties.
    An object is considered array-like if it has attributes typically associated with arrays
    (such as `.shape`, `.dtype`, and `.ndim`), supports indexing, and is iterable.

    Parameters
    ----------
    obj : object
        The object to check for array-like properties.

    Returns
    -------
    bool
        True if the object is array-like, False otherwise.

    Notes
    -----
    This function uses a combination of checks for attributes (`shape`, `dtype`, `ndim`),
    indexability, and iterability to determine if the given object behaves like an array.
    It is designed to be flexible and work with various types of array-like objects, including
    but not limited to NumPy arrays and JAX arrays. However, it may not be full proof for all
    possible array-like types or objects that mimic these properties without being suitable for
    numerical operations.

    """
    # Check for array-like attributes
    has_shape = hasattr(obj, "shape")
    has_dtype = hasattr(obj, "dtype")
    has_ndim = hasattr(obj, "ndim")

    # Check for indexability (try to access the first element)
    try:
        obj[0]
        is_indexable = True
    except (TypeError, IndexError):
        is_indexable = False

    # Check for iterable property
    try:
        iter(obj)
        is_iterable = True
    except TypeError:
        is_iterable = False

    # not_tsd_type = not isinstance(obj, _AbstractTsd)

    return (
        has_shape
        and has_dtype
        and has_ndim
        and is_indexable
        and is_iterable
        # and not_tsd_type
    )


def cast_to_jax(array, array_name, suppress_conversion_warnings=False):
    """
    Convert an input array-like object to a jax Array.


    Parameters
    ----------
    array : array_like
        The input object to convert. This can be any object that `np.asarray` is capable of
        converting to a jax array, such as lists, tuples, and other array-like objects.
    array_name : str
        The name of the variable that we are converting, printed in the warning message.

    Returns
    -------
    ndarray
        A jax Array representation of the input `values`. If `values` is already a jax
        Array, it is returned unchanged. Otherwise, a new jax Array is created and returned.

    Warnings
    --------
    A warning is issued if the input `values` is not already a jax Array, indicating
    that a conversion has taken place and showing the original type of the input.

    """
    if not isinstance(array, jnp.ndarray) and not suppress_conversion_warnings:
        original_type = type(array).__name__
        warnings.warn(
            f"Converting '{array_name}' to jax.ndarray. The provided array was of type '{original_type}'.",
            UserWarning,
            stacklevel=2,
        )
    return jnp.asarray(array)


def convert_to_jax_array(array, array_name, suppress_conversion_warnings=False):
    """Convert any array like object to jax Array.

    Parameters
    ----------
    array : ArrayLike

    array_name : str
        Array name if RuntimeError is raised or object is casted to numpy

    Returns
    -------
    jax.Array
        Jax array object

    Raises
    ------
    RuntimeError
        If input can't be converted to jax array
    """
    if isinstance(array, Number):
        return jnp.array([array])
    elif isinstance(array, (list, tuple)):
        return jnp.array(array)
    elif isinstance(array, jnp.ndarray):
        return array
    elif isinstance(array, np.ndarray):
        return cast_to_jax(array, array_name, suppress_conversion_warnings)
    elif is_array_like(array):
        return cast_to_jax(array, array_name, suppress_conversion_warnings)
    else:
        raise RuntimeError(
            "Unknown format for {}. Accepted formats are numpy.ndarray, list, tuple or any array-like objects.".format(
                array_name
            )
        )


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


def _fill_forward(
    time_target_array,
    time_array,
    data_array,
    starts,
    ends,
    out_of_range=np.nan,
):
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
        (time_target_array.shape[0], *data_array.shape[1:]),
        out_of_range,
        dtype=data_array.dtype,
    )
    fill_idx = 0

    idx_start_target, idx_end_target = _get_idxs(time_target_array, starts, ends)
    idx_start, idx_end = _get_idxs(time_array, starts, ends)

    for i in range(len(idx_start)):
        d = data_array[idx_start[i] : idx_end[i]]
        t = time_array[idx_start[i] : idx_end[i]]
        ts = time_target_array[idx_start_target[i] : idx_end_target[i]]

        idxs = np.searchsorted(t, ts, side="right") - 1
        filled_d[fill_idx : fill_idx + len(ts)][idxs >= 0] = d[idxs[idxs >= 0]]
        fill_idx += len(ts)

    return filled_d


def _get_shifted_indices(idx_start, idx_end, window, start_from=0):
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
    cum_delta = np.concatenate([[start_from], start_from + np.cumsum(idx_end - idx_start)])
    idx_start_shift = window * np.arange(idx_start.shape[0]) + cum_delta[:-1]
    idx_end_shift = window * np.arange(idx_end.shape[0]) + cum_delta[1:]
    return idx_start_shift, idx_end_shift


@jit(nopython=True)
def _get_slicing(idx_start, idx_end):
    """
    Generate an array of indices for slicing, based on provided start and end indices.

    Parameters
    ----------
    idx_start : NDArray
        Array of start indices.
    idx_end : NDArray
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


def _get_complement_slicing(idx_start, idx_end, max_len):
    """
    Generate an array of indices that are not included in the specified slices, based on provided start and end indices.

    Parameters
    ----------
    idx_start : ndarray
        An array of start indices for slicing.
    idx_end : ndarray
        An array of end indices for slicing. Each end index corresponds to a start index.
    max_len : int
        The maximum index to consider (i.e., the length of the array).

    Returns
    -------
    ndarray
        An array of complementary indices that are not covered by the slices defined by `idx_start` and `idx_end`.

    Notes
    -----
    The function computes the indices that fall outside the ranges defined by the start and end indices.
    The result includes:
    1. Indices before the first start index.
    2. Indices between each end index and the subsequent start index.
    3. Indices after the last end index up to `max_len`.

    Examples
    --------
    >>> idx_start = np.array([2, 5, 8])
    >>> idx_end = np.array([4, 6, 10])
    >>> max_len = 12
    >>> _get_complement_slicing(idx_start, idx_end, max_len)
    array([0, 1, 4, 7, 11])
    """
    # Collect complementary indices
    complement_cnt = 0
    slicing_length = np.sum(idx_end - idx_start)
    complement_indices = np.zeros(max_len - slicing_length, dtype=np.int32)

    num_ix = int(idx_start[0] > 0) + int(idx_end[-1] < max_len) + len(idx_start) - 1
    ix_start = np.zeros(num_ix, dtype=np.int32)
    ix_end = np.zeros(num_ix, dtype=np.int32)

    cc_start = 0
    cc_end = 0

    # Add indices before the first start index
    if len(idx_start) and idx_start[0] > 0:

        ix_start[cc_start] = 0
        cc_start += 1
        for k in range(0, idx_start[0]):
            complement_indices[complement_cnt] = k
            complement_cnt += 1

        ix_end[cc_end] = k + 1
        cc_end += 1

    # Add indices between each end and the next start
    for i in range(len(idx_start) - 1):
        if idx_end[i] < idx_start[i + 1]:
            ix_start[cc_start] = idx_end[i]
            cc_start += 1
            for k in range(idx_end[i], idx_start[i + 1]):
                complement_indices[complement_cnt] = k
                complement_cnt += 1
            ix_end[cc_end] = k + 1
            cc_end += 1

    # Add indices after the last end index
    if len(idx_end) > 0 and idx_end[-1] < max_len:
        ix_start[cc_start] = idx_end[-1]

        for k in range(idx_end[-1], max_len):
            complement_indices[complement_cnt] = k
            complement_cnt += 1
        ix_end[cc_end] = k + 1

    return complement_indices, ix_start, ix_end


@jit(nopython=True)
def _revert_epochs(idx_start, idx_end):
    """
    Generate an array of indices for reversing data order within each epoch.

    Parameters
    ----------
    idx_start: NDArray
        Start epoch indices.
    idx_end: NDArray
        End epoch indices.

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
            ix[cnt] = en + st - k - 1
            cnt += 1
    return ix


def pad_and_roll(count_array, windows, constant_value=np.nan):
    """
    Pad and roll the input array to generate shifted versions of the array according
    to specified window size and padding direction.

    Parameters
    ----------
    count_array : ArrayLike
        The input array to pad and roll. This is typically a count or spike array in
        neural data analysis.
    windows : tuple of int
        The number of steps to include in the window. This defines the extent of the
        rolling operation.
    constant_value: float
        Padding constant

    Returns
    -------
    ArrayLike
        A 2D array where each row represents the input array rolled by one step in
        the range defined by the window and padding type. Only the valid range (original
        data indices) is returned.

    Notes
    -----
    The function uses `np.nan` for padding, which may need to be considered in subsequent
    calculations. Depending on the analysis, handling of `np.nan` may be required to avoid
    statistical or computational errors.
    """
    n_samples = count_array.shape[0]
    pad = lambda x: jnp.pad(
        x, pad_width=(windows, (0, 0)), constant_values=constant_value
    )
    indices = jnp.arange(-windows[0], windows[1] + 1)[::-1]
    idx = jnp.arange(windows[0], n_samples + windows[0])
    roll = jax.vmap(lambda i: jnp.roll(pad(count_array), -i, axis=0))
    return roll(indices)[:, idx]


def _odd_ext_multiepoch(n_pts, time, data, starts, ends):
    """
    Apply an odd extension at the boundaries of a multi-epoch data array using JAX.

    This function extends the data array at both ends using an odd extension (i.e., mirroring the border values
    with a decrement of the same step). It supports both single and multiple epochs.

    Parameters
    ----------
    n_pts : int
        The number of points by which to extend the data at each end of the array for each epoch.
    time : jnp.ndarray
        The time array corresponding to the data, used to determine the start and end indices for each epoch.
    data : jnp.ndarray
        The input data array to be extended, with shape (n_samples, ...).
    starts : jnp.ndarray
        Array of start times for each epoch.
    ends : jnp.ndarray
        Array of end times for each epoch.

    Returns
    -------
    ext : jnp.ndarray
        The extended data array with odd extensions at the boundaries. The shape will be larger than the input
        data array due to the added padding.
    ix_pad_start : jnp.ndarray
        Array of start indices for the padding regions.
    ix_pad_end : jnp.ndarray
        Array of end indices for the padding regions.
    ix_data : jnp.ndarray
        Array of indices corresponding to the original data positions within the extended array.

    Raises
    ------
    ValueError
        If the length of the data per each epoch is not sufficient to perform the required padding.

    Examples
    --------
    >>> n_pts = 3
    >>> time = jnp.array([0, 1, 2, 3, 4, 5, 6])
    >>> data = jnp.array([10, 20, 30, 40, 50, 60, 70])
    >>> starts = jnp.array([1])
    >>> ends = jnp.array([5])
    >>> ext, ix_pad_start, ix_pad_end, ix_data = _odd_ext_multiepoch(n_pts, time, data, starts, ends)
    >>> print(ext)
    # Output: Extended array with odd extensions at the boundaries

    Notes
    -----
    The function supports both single and multi-epoch data. For single epochs, it performs a simple odd extension
    at the boundaries. For multi-epoch data, it slices the data into epochs, applies odd extension to each epoch,
    and then reconstructs the extended array. The function ensures that `n_pts` is smaller than the length of
    the data for each epoch.
    """
    if len(starts) <= 1:

        if data.shape[0] < n_pts:
            raise ValueError("The length of the data per each epoch must be greater "
                             f"than `{n_pts}`, which is required by the padding used "
                             "in this algorithm.")

        ext = jnp.concatenate(
            (
                2 * data[0] - data[slice(n_pts, 0, -1)],
                data,
                2 * data[-1] - data[slice(-2, -(n_pts + 2), -1)]
            ),
            axis=0
        )
        ix_pad_start, idx_pad_end = jnp.array([0]), jnp.array([len(ext)])
        ix_data = jnp.arange(n_pts, data.shape[0] + n_pts)
    else:
        # check epoch duration
        idx_start = np.searchsorted(time, starts)
        idx_end = np.searchsorted(time, ends, side="right")

        if np.any(idx_end - idx_start < n_pts):
            raise ValueError("The length of the data per each epoch must be greater "
                             f"than `{n_pts}`, which is required by the padding used "
                             "in this algorithm.")
        # use slower slicing with indices with multi-epochs
        # use broadcasting trick
        i_after_start = idx_start[:, jnp.newaxis] + jnp.ones((len(idx_start), 1), dtype=int) * jnp.arange(n_pts, 0, -1)[jnp.newaxis]
        i_before_end = idx_end[:, jnp.newaxis] - jnp.ones((len(idx_start), 1), dtype=int) * jnp.arange(2, n_pts + 2)[jnp.newaxis]

        # calculate edges as the above, more readable formula and store it
        # uses broadcasting
        edges = np.hstack(
            (
                2 * data[idx_start, jnp.newaxis] - data[i_after_start],
                2 * data[idx_end-1, jnp.newaxis] - data[i_before_end])
        ).reshape(-1, *data.shape[1:])

        # count how many time points the padded array has
        new_size = data.shape[0] + edges.shape[0]

        # get the indices for setting elements
        # shift by a window every epoch
        idx_start_shift, idx_end_shift = _get_shifted_indices(
            idx_start, idx_end, 2*n_pts, start_from=n_pts
        )
        ix_data = _get_slicing(idx_start_shift, idx_end_shift)
        ix_padding, ix_pad_start, idx_pad_end = _get_complement_slicing(idx_start_shift, idx_end_shift, new_size)

        # instantiate new array
        ext = jnp.zeros((new_size, *data.shape[1:])).at[ix_data].set(data).at[ix_padding].set(edges)

    return ext, ix_pad_start, idx_pad_end, ix_data
