from typing import Union

import jax
import jax.numpy as jnp
import numba
import numpy as np
import pynapple as nap

from .utils import _get_idxs, _fill_forward, _get_shifted_indices, _get_slicing




def pad_and_roll(count_array, windows):
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
    pad = lambda x: jnp.pad(
        x, pad_width=(windows, (0, 0)), constant_values=np.nan
    )
    indices = jnp.arange(-windows[0], windows[1] + 1)[::-1]
    idx = jnp.arange(windows[0], n_samples + windows[1] + 2)
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


def sta_single_epoch(count_array, data_array, windows, batch_size=256):
    """
    Compute the spike-triggered average (STA) for a single epoch.

    Parameters
    ----------
    count_array : ArrayLike
        Array of spike counts.
    data_array : ArrayLike
        Array of corresponding data/stimulus values.
    windows : tuple of int
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
    shape = data_array.shape
    data_array = data_array.reshape(data_array.shape[0], -1)
    rolled_counts = pad_and_roll(count_array, windows)
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




# def compute_average(
#     time_array,
#     count_array,
#     time_array2,
#     data_array,
#     starts,
#     ends,
#     window,
#     batch_size=256,
#     side="backward",
#     ):
#     count_array = jnp.asarray(count_array, dtype=float)
#     data_array = jnp.asarray(data_array, dtype=float)

#     idx_start, idx_end = _get_idxs(time_array, starts, ends)
#     # need numpy
#     idx_start_shift, idx_end_shift = _get_shifted_indices(idx_start, idx_end, window)

#     # get the indices for setting elements
#     ix_orig = _get_slicing(idx_start, idx_end)
#     ix_shift = _get_slicing(idx_start_shift, idx_end_shift)

#     # define larger array
#     tot_size = ix_shift[-1] - ix_shift[0] + 1

#     if batch_size >= data_array[0].size:
#         # add nans between trials
#         data_array = (
#             jnp.full((tot_size, *data_array.shape[1:]), np.nan)
#             .at[ix_shift]
#             .set(data_array[ix_orig])
#         )
#         count_array = (
#             jnp.full((tot_size, *count_array.shape[1:]), np.nan)
#             .at[ix_shift]
#             .set(count_array[ix_orig])
#         )
#         # compute a single epoch sta
#         res = sta_single_epoch(
#             count_array, data_array, window, batch_size=batch_size, side=side
#         )
#     else:
#         # batch features
#         res = _batched_dot_prod_feature_multi(
#             tot_size,
#             ix_orig,
#             ix_shift,
#             count_array,
#             data_array,
#             window,
#             batch_size,
#             side,
#         )

#     return res


def event_trigger_average(time_target_array,count_array,time_array,data_array,starts,ends,windows,binsize,batch_size):
    """
    Main function to call for event-triggered averages (ETA) across multiple epochs.
    This function assumes count array and data array have been restricted before.

    Parameters
    ----------
    time_target_array : ArrayLike
        Timestamps array of counts
    count_array : ArrayLike
        Count array of events
    time_array : ArrayLike
        Timestamps of data. Can be lower or above sampling rate of time_array
    data_array : ArrayLike
        Data to average at event
    starts : ArrayLike
        Start times of the epochs over which the STA is computed.
    ends : ArrayLike
        End times of the epochs, corresponding to each start time.
    windows : tuple of int
        The number of time steps to include before and after each event for averaging.
    binsize : float
        The bin size. Used by `bin_average`.
    batch_size : int, optional
        Number of elements to process in each batch.

    Returns
    -------
    jnp.ndarray :
        The combined STA calculated across all specified epochs.

    """

    # Need to bring data_target_array to same shape as count_array

    # bin_average
    if count_array.shape[0] < data_target_array[0] : 
        data_array = bin_average(time_array, data_array, starts, ends, binsize)
    # fill_forward
    else: 
        data_array = _fill_forward(time_target_array, time_array, data_array, starts, ends)

    idx_start, idx_end = _get_idxs(time_target_array, starts, ends)
    idx_start_shift, idx_end_shift = _get_shifted_indices(idx_start, idx_end, np.sum(windows)+1)

    # get the indices for setting elements
    ix_orig = _get_slicing(idx_start, idx_end)
    ix_shift = _get_slicing(idx_start_shift, idx_end_shift)

    # define larger array
    tot_size = ix_shift[-1] - ix_shift[0] + 1

    shape = data_array.shape
    data_array = data_array.reshape(shape[0], -1)
    if len(shape) == 1:
        data_array = data_array.T
    
    num_full_batches = data_array.shape[1] // batch_size
    carry = 0

    count_array = (
        jnp.full((tot_size, *count_array.shape[1:]), np.nan)
        .at[ix_shift]
        .set(count_array[ix_orig])
    )
    count_array = pad_and_roll(count_array, windows)

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

    # if batch_size >= np.prod(data_array.shape[1:]):
    #     # add nans between trials
    #     data_array = (
    #         jnp.full((tot_size, *data_array.shape[1:]), np.nan)
    #         .at[ix_shift]
    #         .set(data_array[ix_orig])
    #     )
    #     count_array = (
    #         jnp.full((tot_size, *count_array.shape[1:]), np.nan)
    #         .at[ix_shift]
    #         .set(count_array[ix_orig])
    #     )
    #     # compute a single epoch sta
    #     res = sta_single_epoch(
    #         count_array, data_array, window, batch_size=batch_size, side=side
    #     )
    # else:
    #     # batch features
    #     res = _batched_dot_prod_feature_multi(
    #         tot_size,
    #         ix_orig,
    #         ix_shift,
    #         count_array,
    #         data_array,
    #         window,
    #         batch_size,
    #         side,
    #     )

    # return res    