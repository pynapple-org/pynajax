from typing import Union

import jax
import jax.numpy as jnp
import numba
import numpy as np
import pynapple as nap

from .utils import _get_idxs

TsdType = Union[nap.Tsd, nap.TsdFrame, nap.TsdTensor]


def fill_forward(time_series, data, ep=None, out_of_range=np.nan):
    """
    Fill a time series forward in time with data.

    Parameters
    ----------
    time_series:
        The time series to match.
    data: Tsd, TsdFrame, or TsdTensor
        The time series with data to be extend.

    Returns
    -------
    : Tsd, TsdFrame, or TsdTensor
        The data time series filled forward.

    Raises
    ------
    AssertionError
        If `data` is not an instance of TsdType or `ep` is not None and not an instance of IntervalSet.

    """
    assert isinstance(data, TsdType)

    if ep is None:
        ep = time_series.time_support
    else:
        assert isinstance(ep, nap.IntervalSet)
        time_series.restrict(ep)

    data = data.restrict(ep)
    starts = ep.start
    ends = ep.end

    filled_d = np.full(
        (time_series.t.shape[0], *data.shape[1:]), out_of_range, dtype=data.dtype
    )
    fill_idx = 0
    for start, end in zip(starts, ends):
        data_ep = data.get(start, end)
        ts_ep = time_series.get(start, end)
        idxs = np.searchsorted(data_ep.t, ts_ep.t, side="right") - 1
        filled_d[fill_idx : fill_idx + ts_ep.t.shape[0]][idxs >= 0] = data_ep.d[
            idxs[idxs >= 0]
        ]
        fill_idx += ts_ep.t.shape[0]
    return type(data)(t=time_series.t, d=filled_d, time_support=ep)


def pad_and_roll(count_array, window, padding_type="backward"):
    """
    Pad and roll the input array to generate shifted versions of the array according
    to specified window size and padding direction.

    Parameters
    ----------
    count_array : ArrayLike
        The input array to pad and roll. This is typically a count or spike array in
        neural data analysis.
    window : int
        The number of steps to include in the window. This defines the extent of the
        rolling operation.
    padding_type : Literal["backward", "forward", "both"], optional
        The type of padding to apply. Can be 'backward' for padding past values,
        'forward' for future values, or 'both' for both past and future. Default is 'backward'.

    Returns
    -------
    ArrayLike
        A 2D array where each row represents the input array rolled by one step in
        the range defined by the window and padding type. Only the valid range (original
        data indices) is returned.

    Raises
    ------
    ValueError
        If an invalid `padding_type` is provided.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> count_array = jnp.array([1., 2., 3., 4., 5., 6.]).reshape(3, 2)
    >>> window = 2
    >>> pad_and_roll(count_array, window, 'backward')
    >>> pad_and_roll(count_array, window, 'forward')
    >>> pad_and_roll(count_array, window, 'both')

    Notes
    -----
    The function uses `np.nan` for padding, which may need to be considered in subsequent
    calculations. Depending on the analysis, handling of `np.nan` may be required to avoid
    statistical or computational errors.
    """
    n_samples = count_array.shape[0]
    if padding_type == "backward":
        pad = lambda x: jnp.pad(
            x, pad_width=((0, window), (0, 0)), constant_values=np.nan
        )
        indices = jnp.arange(window, -1, -1)
        idx = jnp.arange(0, n_samples)
    elif padding_type == "forward":
        pad = lambda x: jnp.pad(
            x, pad_width=((window, 0), (0, 0)), constant_values=np.nan
        )
        indices = -jnp.arange(0, window + 1)
        idx = jnp.arange(window, n_samples + window)
    elif padding_type == "both":
        pad = lambda x: jnp.pad(
            x, pad_width=((window, window), (0, 0)), constant_values=np.nan
        )
        indices = jnp.arange(-window, window + 1)[::-1]
        idx = jnp.arange(window, n_samples + window)
    else:
        raise ValueError("Invalid padding_type. Use 'backward', 'forward', or 'both'.")

    roll = jax.vmap(lambda i: jnp.roll(pad(count_array), -i, axis=0))
    return roll(indices)[:, idx]


# dot prod shifted counts vs 1D var y vmapped over shift
# [(n_shift, T), (T, )] -> (n_shift, )
_dot_prod = jax.vmap(lambda x, y: jnp.nanmean(x * y), in_axes=(0, None), out_axes=0)
# vmap over the neurons
# [(n_shift, T, n_neurons), (T, )] -> (n_shift, n_neurons)
_dot_prod_neu = jax.vmap(_dot_prod, in_axes=(2, None), out_axes=1)
# vmap over the features
# [(n_shift, T, n_neurons), (T, n_features)] -> (n_shift, n_neurons, n_features)
_dot_prod_feature = jax.vmap(_dot_prod_neu, in_axes=(None, 1), out_axes=2)


def _dot_prod_feature_and_reshape(data_array, count_array, window, shape, side):
    """Apply the dot product and reshape to (n_shifts, n_neuron, *shape[1:])."""
    res = _dot_prod_feature(count_array, data_array)
    return _reshape_sta(res, window, count_array.shape[2], shape, side)


def _batched_dot_prod_feature(data_array, count_array, batch_size, window, shape, side):
    """
    Compute the dot product of shifted count arrays and a data array in batches, reshaping the output
    to accommodate features across the specified window with a side offset.

    Parameters
    ----------
    data_array : ArrayLike
        The array of data values.
    count_array : ArrayLike
        The array of counts to be shifted and used in the dot product computation.
    batch_size : int
        The number of samples to include in each batch during computation.
    window : int
        The number of steps to shift for the rolling operation.
    shape : tuple
        The shape to reshape the final results into.
    side : {'backward', 'forward', 'both'}
        Specifies which side of the padding to use when rolling the counts.

    Returns
    -------
    ArrayLike :
        The reshaped array of dot product results, adjusted according to the specified window
        and side settings.
    """
    num_full_batches = data_array.shape[1] // batch_size
    carry = 0

    def scan_fn(carry, x):
        slc = jax.lax.dynamic_slice(
            data_array, (0, carry), (data_array.shape[0], batch_size)
        )
        batch_result = _dot_prod_feature(count_array, slc)
        return carry + batch_size, batch_result

    _, res = jax.lax.scan(scan_fn, carry, None, length=num_full_batches)

    res = jnp.transpose(res, (1, 2, 0, 3))  # move features at the end
    res = res.reshape(*res.shape[:-2], -1)  # flatten features

    extra_elements = data_array.shape[1] % batch_size
    if extra_elements:
        # compute residual slice
        resid = _dot_prod_feature(count_array, data_array[:, -extra_elements:])
        resid = resid.transpose(1, 2, 0).reshape(*res.shape[:-1], -1)
        res = np.concatenate([res, resid], axis=2)

    # reshape back to original and return
    return _reshape_sta(res, window, count_array.shape[2], shape, side)


def _reshape_sta(sta, window, n_neurons, shape, side):
    """
   Reshape the spike-triggered average (STA) results according to the specified window and padding configuration.

   Parameters
   ----------
   sta : ArrayLike
       The spike-triggered average data that needs to be reshaped.
   window : int
       The number of steps to include in the window. This defines the extent of the STA calculation.
   n_neurons : int
       The number of neurons (or channels) in the STA data.
   shape : tuple
       The shape of the original data array, used to format the output shape.
   side : {'backward', 'forward', 'both'}
       Specifies the padding side used in the STA calculation, affecting the output shape.

   Returns
   -------
   jnp.ndarray :
       The reshaped STA array, formatted according to the input specifications.
   """
    if side == "both":
        return sta.reshape((2 * window + 1, n_neurons, *shape[1:]))
    return sta.reshape((window + 1, n_neurons, *shape[1:]))


def sta_single_epoch(count_array, data_array, window, batch_size=256, side="backward"):
    """
    Compute the spike-triggered average (STA) for a single epoch.

    Parameters
    ----------
    count_array : ArrayLike
        Array of spike counts.
    data_array : ArrayLike
        Array of corresponding data/stimulus values.
    window : int
        The number of time steps to include before and after each spike for averaging.
    batch_size : int, optional
        The number of samples to include in each batch for processing. Defaults to 256.
    side : {'backward', 'forward', 'both'}, optional
        Specifies which side of the padding to use when rolling the counts. Defaults to 'backward'.

    Returns
    -------
    jnp.ndarray :
        The computed STA array for the single epoch.

    Examples
    --------
    >>> count_array = jnp.array([...])
    >>> data_array = jnp.array([...])
    >>> sta_result = sta_single_epoch(count_array, data_array, 10, 256, 'backward')
    """
    count_array = jnp.asarray(count_array, dtype=float)
    data_array = jnp.asarray(data_array, dtype=float)
    shape = data_array.shape
    data_array = data_array.reshape(data_array.shape[0], -1)
    rolled_counts = pad_and_roll(count_array, window, padding_type=side)
    if batch_size >= data_array.shape[1]:
        res = _dot_prod_feature_and_reshape(
            data_array, rolled_counts, window, shape, side
        )
    else:
        res = _batched_dot_prod_feature(
            data_array, rolled_counts, batch_size, window, shape, side
        )
    return res


def sta_multi_epoch_loop(
    time_array,
    starts,
    ends,
    count_array,
    data_array,
    window,
    batch_size=256,
    side="backward",
):
    """
    Compute a multi-epoch STA with a loop over epochs.

    Parameters
    ----------
    time_array
    starts
    ends
    count_array
    data_array
    window
    batch_size

    Returns
    -------

    """
    idx_start, idx_end = _get_idxs(time_array, starts, ends)

    tree_data = [data_array[start:end] for start, end in zip(idx_start, idx_end)]
    tree_count = [count_array[start:end] for start, end in zip(idx_start, idx_end)]

    def sta(cnt, dat):
        results = sta_single_epoch(cnt, dat, window, batch_size=batch_size, side=side)
        return results

    delta = idx_end - jnp.arange(window + 1)[:, jnp.newaxis] - idx_start
    if side == "both":
        delta = jnp.vstack((delta[1:][::-1], delta))

    frac_duration = delta / jnp.sum(delta, axis=1)[:, np.newaxis]
    frac_duration = frac_duration[:, :, np.newaxis, np.newaxis, np.newaxis]
    res = sum(
        [
            sta_res * frac_duration[:, k]
            for k, sta_res in enumerate(jax.tree_map(sta, tree_count, tree_data))
        ]
    )

    return res


def _get_shifted_indices(i_start, i_end, window):
    """
    Compute shifted indices for given start and end indices based on a specified window size.

    Parameters
    ----------
    i_start : NDArray
        Array of start indices.
    i_end : NDArray
        Array of end indices.
    window : int
        The number of units each index should be shifted to create a windowed effect.

    Returns
    -------
    : tuple of NDArray
        A tuple containing two arrays: shifted start indices and shifted end indices.
    """
    cum_delta = np.concatenate([[0], np.cumsum(i_end - i_start)])
    i_start_shift = window * np.arange(i_start.shape[0]) + cum_delta[:-1]
    i_end_shift = window * np.arange(i_end.shape[0]) + cum_delta[1:]
    return i_start_shift, i_end_shift


@numba.njit
def _get_slicing(i_start, i_end):
    """
    Generate an array of indices for slicing, based on provided start and end indices.

    Parameters
    ----------
    i_start : NDArray
        Array of start indices.
    i_end : NDArray
        Array of end indices.

    Returns
    -------
    : NDArray
        An array of indices constructed from the start and end indices.
    """
    iend = np.sum(i_end - i_start)
    ix = np.zeros(iend, dtype=np.int32)
    cnt = 0
    for st, en in zip(i_start, i_end):
        for k in range(st, en):
            ix[cnt] = k
            cnt += 1
    return ix


def sta_multi_epoch(
    time_array,
    starts,
    ends,
    count_array,
    data_array,
    window,
    batch_size=256,
    side="backward",
):
    """
    Compute spike-triggered averages (STA) across multiple epochs.

    Parameters
    ----------
    time_array : ArrayLike
        Array of timestamps defining the overall time context for the analysis.
    starts : ArrayLike
        Start times of the epochs over which the STA is computed.
    ends : ArrayLike
        End times of the epochs, corresponding to each start time.
    count_array : ArrayLike
        Array of counts or spikes, used to compute the STA.
    data_array : ArrayLike
        Array of data values associated with each timestamp in `time_array`.
    window : int
        The number of time steps to include before and after each spike for averaging.
    batch_size : int, optional
        Number of elements to process in each batch.
    side : str, optional
        Specifies whether the window should look 'backward', 'forward', or 'both'.

    Returns
    -------
    jnp.ndarray :
        The combined STA calculated across all specified epochs.

    Examples
    --------
    >>> time_array = np.array([...])
    >>> starts = np.array([...])
    >>> ends = np.array([...])
    >>> count_array = np.array([...])
    >>> data_array = np.array([...])
    >>> sta_result = sta_multi_epoch(time_array, starts, ends, count_array, data_array, 10)
    """
    count_array = jnp.asarray(count_array, dtype=float)
    data_array = jnp.asarray(data_array, dtype=float)

    idx_start, idx_end = _get_idxs(time_array, starts, ends)
    # need numpy
    idx_start_shift, idx_end_shift = _get_shifted_indices(idx_start, idx_end, window)

    # get the indices for setting elements
    ix_orig = _get_slicing(idx_start, idx_end)
    ix_shift = _get_slicing(idx_start_shift, idx_end_shift)

    # define larger array
    tot_size = ix_shift[-1] - ix_shift[0] + 1

    if batch_size >= data_array[0].size:
        # add nans between trials
        data_array = (
            jnp.full((tot_size, *data_array.shape[1:]), np.nan)
            .at[ix_shift]
            .set(data_array[ix_orig])
        )
        count_array = (
            jnp.full((tot_size, *count_array.shape[1:]), np.nan)
            .at[ix_shift]
            .set(count_array[ix_orig])
        )
        # compute a single epoch sta
        res = sta_single_epoch(
            count_array, data_array, window, batch_size=batch_size, side=side
        )
    else:
        # batch features
        res = _batched_dot_prod_feature_multi(
            tot_size,
            ix_orig,
            ix_shift,
            count_array,
            data_array,
            window,
            batch_size,
            side,
        )

    return res


def _batched_dot_prod_feature_multi(
    tot_size, ix_orig, ix_shift, count_array, data_array, window, batch_size, side
):
    """
    Perform batched dot product calculations for feature arrays over multiple epochs.

    Parameters
    ----------
    tot_size : int
        The total size of the array to accommodate all batches.
    ix_orig : NDArray
        Original indices to reference data and count arrays.
    ix_shift : NDArray
        Shifted indices to align data for the rolling operation.
    count_array : jnp.ndarray
        Array of spike counts.
    data_array : jnp.ndarray
        Data array corresponding to spike counts.
    window : int
        Window size for rolling count data.
    batch_size : int
        Size of each batch for processing.
    side : str
        Specifies whether to roll counts 'backward', 'forward', or 'both' for STA calculation.

    Returns
    -------
    jnp.ndarray :
        The result of the batched dot product feature computation.
    """
    # get shape and reshape data
    shape = data_array.shape
    data_array = data_array.reshape(shape[0], -1)

    num_full_batches = data_array.shape[1] // batch_size
    carry = 0

    count_array = (
        jnp.full((tot_size, *count_array.shape[1:]), np.nan)
        .at[ix_shift]
        .set(count_array[ix_orig])
    )
    count_array = pad_and_roll(count_array, window, padding_type=side)

    def scan_fn(carry, x):
        slc = jax.lax.dynamic_slice(
            data_array, (0, carry), (data_array.shape[0], batch_size)
        )
        slc = (
            jnp.full((tot_size, *slc.shape[1:]), np.nan).at[ix_shift].set(slc[ix_orig])
        )
        batch_result = _dot_prod_feature(count_array, slc)
        return carry + batch_size, batch_result

    _, res = jax.lax.scan(scan_fn, carry, None, length=num_full_batches)

    res = jnp.transpose(res, (1, 2, 0, 3))  # move features at the end
    res = res.reshape(*res.shape[:-2], -1)  # flatten features

    extra_elements = data_array.shape[1] % batch_size
    if extra_elements:
        # compute residual slice
        slc = data_array[:, -extra_elements:]
        slc = (
            jnp.full((tot_size, *slc.shape[1:]), np.nan).at[ix_shift].set(slc[ix_orig])
        )

        resid = _dot_prod_feature(count_array, slc)
        resid = resid.transpose(1, 2, 0).reshape(*res.shape[:-1], -1)
        res = np.concatenate([res, resid], axis=2)

    # reshape back to original and return
    return _reshape_sta(res, window, count_array.shape[2], shape, side)
