import jax
import jax.numpy as jnp
import numpy as np

from pynajax.jax_core_bin_average import bin_average

from .utils import _fill_forward, _get_idxs, _get_shifted_indices, _get_slicing

# dot prod shifted counts vs 1D var y vmapped over shift
# [(n_shift, T), (T, )] -> (n_shift, )
_dot_prod = jax.vmap(lambda x, y: jnp.nansum(x * y), in_axes=(0, None), out_axes=0)
# vmap over the neurons
# [(n_shift, T, n_neurons), (T, )] -> (n_shift, n_neurons)
_dot_prod_neu = jax.vmap(_dot_prod, in_axes=(2, None), out_axes=1)
# vmap over the features
# [(n_shift, T, n_neurons), (T, n_features)] -> (n_shift, n_neurons, n_features)
_dot_prod_feature = jax.vmap(_dot_prod_neu, in_axes=(None, 1), out_axes=2)


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
    pad = lambda x: jnp.pad(x, pad_width=(windows, (0, 0)), constant_values=np.nan)
    indices = jnp.arange(-windows[0], windows[1] + 1)[::-1]
    idx = jnp.arange(windows[0], n_samples + windows[0])
    roll = jax.vmap(lambda i: jnp.roll(pad(count_array), -i, axis=0))
    return roll(indices)[:, idx]


def event_trigger_average(
    time_target_array,
    count_array,
    time_array,
    data_array,
    starts,
    ends,
    windows,
    binsize,
    batch_size,
):
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
    if count_array.shape[0] < data_array.shape[0]:
        time_array, data_array = bin_average(time_array, data_array, starts, ends, binsize)
    # fill_forward
    else:
        data_array = _fill_forward(time_target_array, time_array, data_array, starts, ends)

    idx_start, idx_end = _get_idxs(time_target_array, starts, ends)
    idx_start_shift, idx_end_shift = _get_shifted_indices(idx_start, idx_end, np.sum(windows) + 1)

    # get the indices for setting elements
    ix_orig = _get_slicing(idx_start, idx_end)
    ix_shift = _get_slicing(idx_start_shift, idx_end_shift)

    # define larger array
    tot_size = ix_shift[-1] - ix_shift[0] + 1

    shape = data_array.shape
    if data_array.ndim == 1:
        data_array = np.expand_dims(data_array, -1)
    else:
        data_array = data_array.reshape(shape[0], -1)

    num_full_batches = np.maximum(1, data_array.shape[1] // batch_size)
    batch_size = np.minimum(data_array.shape[1], batch_size)
    carry = 0

    tot_count = jnp.sum(count_array, 0)

    count_array = (
        jnp.full((tot_size, *count_array.shape[1:]), np.nan).at[ix_shift].set(count_array[ix_orig])
    )
    count_array = pad_and_roll(count_array, windows)

    def scan_fn(carry, x):
        slc = jax.lax.dynamic_slice(data_array, (0, carry), (data_array.shape[0], batch_size))
        slc = jnp.full((tot_size, *slc.shape[1:]), np.nan).at[ix_shift].set(slc[ix_orig])
        batch_result = _dot_prod_feature(count_array, slc)
        return carry + batch_size, batch_result

    _, res = jax.lax.scan(scan_fn, carry, None, length=num_full_batches)

    res = jnp.transpose(res, (1, 2, 0, 3))  # move features at the end
    res = res.reshape(*res.shape[:-2], -1)  # flatten features

    extra_elements = data_array.shape[1] % batch_size
    if extra_elements:
        # compute residual slice
        slc = data_array[:, -extra_elements:]
        slc = jnp.full((tot_size, *slc.shape[1:]), np.nan).at[ix_shift].set(slc[ix_orig])

        resid = _dot_prod_feature(count_array, slc)
        resid = resid.transpose(1, 2, 0).reshape(*res.shape[:-1], -1)
        res = np.concatenate([res, resid], axis=2)

    # reshape back to original
    res = res.reshape((np.sum(windows) + 1, count_array.shape[-1], *shape[1:]))

    res = jnp.apply_along_axis(jnp.divide, 1, res, tot_count)

    return res
