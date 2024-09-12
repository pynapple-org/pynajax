from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal as signal

from .utils import (
    _get_shifted_indices,
    _get_slicing,
    _odd_ext_multiepoch,
    _revert_epochs,
)


@partial(jax.jit, static_argnums=(3, ))
def _recursion_loop_sos(signal, sos, zi, nan_function):
    """
    Applies a recursive second-order section (SOS) filter to the input signal.

    Parameters
    ----------
    signal : jnp.ndarray
        The input signal to be filtered, with shape (n_samples,).
    sos : jnp.ndarray
        Array of second-order filter coefficients in the 'sos' format, with shape (n_sections, 6).
    zi : jnp.ndarray
        Initial conditions for the filter, with shape (n_sections, 2, n_epochs).
    nan_function : callable
        A function that specifies how to re-initialize the initial conditions when a NaN is encountered in the signal.
        It should take two arguments: the epoch number and the current filter state, and return a tuple of the updated
        epoch number and the re-initialized filter state.

    Returns
    -------
    jnp.ndarray
        The filtered signal, with the same shape as the input signal.
    """

    def internal_loop(s, x_zi):
        x_cur, zi_slice = x_zi
        x_new = sos[s, 0] * x_cur + zi_slice[s, 0]
        zi_slice = zi_slice.at[s, 0].set(
            sos[s, 1] * x_cur - sos[s, 4] * x_new + zi_slice[s, 1]
        )
        zi_slice = zi_slice.at[s, 1].set(
            sos[s, 2] * x_cur - sos[s, 5] * x_new)
        x_cur = x_new
        return x_cur, zi_slice

    def recursion_step(carry, x):
        epoch_num, zi_slice = carry

        x_cur, zi_slice = jax.lax.fori_loop(
            lower=0, upper=sos.shape[0], body_fun=internal_loop, init_val=(x, zi_slice)
        )

        # Use jax.lax.cond to choose between nan_case and not_nan_case
        epoch_num, zi_slice = jax.lax.cond(
            jnp.isnan(x),  # Condition to check
            nan_function,  # Function to call if x is NaN
            lambda i, x: (i, zi_slice),  # Function to call if x is not NaN
            epoch_num,
            zi,
        )

        return (epoch_num, zi_slice), x_cur

    _, res = jax.lax.scan(recursion_step, (0, zi[..., 0]), signal)

    return res


# vectorize the recursion over signals.
_vmap_recursion_sos = jax.vmap(_recursion_loop_sos, in_axes=(1, None, 2, None), out_axes=1)


def _insert_constant(idx_start, idx_end, data_array, window_size, const=jnp.nan):
    """
    Insert a constant value array between epochs in a time series data array.

    This function interleaves a constant value array of specified size between each epoch in the data array.

    Parameters
    ----------
    idx_start : jnp.ndarray
        Array of start indices for each epoch.
    idx_end : jnp.ndarray
        Array of end indices for each epoch.
    data_array : jnp.ndarray
        The input data array, with shape (n_samples, ...).
    window_size : int
        The size of the constant array to be inserted between epochs.
    const : float, optional
        The constant value to be inserted, by default jnp.nan.

    Returns
    -------
    data_array: jnp.ndarray
        The modified data array with the constant arrays inserted.
    ix_orig: jnp.ndarray
        Indices corresponding to the samples in the original data array.
    ix_shift: jnp.ndarray
        The shifted indices after the constant array has been interleaved.
    idx_start_shift:
        The shifted start indices of the epochs in the modified array.
    idx_end_shift:
        The shifted end indices of the epochs in the modified array.
    """
    # shift by a window every epoch
    idx_start_shift, idx_end_shift = _get_shifted_indices(
        idx_start, idx_end, window_size
    )

    # get the indices for setting elements
    ix_orig = _get_slicing(idx_start, idx_end)
    ix_shift = _get_slicing(idx_start_shift, idx_end_shift)

    tot_size = ix_shift[-1] - ix_shift[0] + 1
    data_array = (
        jnp.full((tot_size, *data_array.shape[1:]), const)
        .at[ix_shift]
        .set(data_array[ix_orig])
    )
    return data_array, ix_orig, ix_shift, idx_start_shift, idx_end_shift


def jax_sosfiltfilt(sos, time_array, data_array, starts, ends):
    """
    Apply forward-backward filtering using a second-order section (SOS) filter.

    This function applies an SOS filter to the data array in both forward and reverse directions,
    which results in zero-phase filtering.

    Parameters
    ----------
    sos : np.ndarray
        Array of second-order filter coefficients in the 'sos' format, with shape (n_sections, 6).
    time_array : np.ndarray
        The time array corresponding to the data, with shape (n_samples,).
    data_array : jnp.ndarray
        The data array to be filtered, with shape (n_samples, ...).
    starts : np.ndarray
        Array of start indices for the epochs in the data array.
    ends : np.ndarray
        Array of end indices for the epochs in the data array.

    Returns
    -------
    : jnp.ndarray
        The zero-phase filtered data array, with the same shape as the input data array.
    """

    original_shape = data_array.shape
    data_array = data_array.reshape(data_array.shape[0], -1)

    # same default padding as scipy.sosfiltfilt ("pad" method and "odd" padtype).
    n_sections = sos.shape[0]
    ntaps = 2 * n_sections + 1
    ntaps -= min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    pad_num = 3 * ntaps

    ext, ix_start_pad, ix_end_pad, ix_data = _odd_ext_multiepoch(pad_num, time_array, data_array, starts, ends)

    # get the start/end index of each epoch after padding
    ix_start_ep = np.hstack((ix_start_pad[0], ix_start_pad[1:-1] + pad_num))
    ix_end_ep = np.hstack((ix_start_ep[1:], ix_end_pad[-1]))

    zi = signal.sosfilt_zi(sos)

    # this braodcast has shape (*zi.shape, data_array.shape[1], len(ix_start_pad))
    z0 = zi[..., jnp.newaxis, jnp.newaxis] * ext.T[jnp.newaxis, jnp.newaxis, ..., ix_start_ep]

    if len(starts) > 1:
        # multi epoch case augmenting with nans.
        aug_data, ix_orig, ix_shift, idx_start_shift, idx_end_shift = _insert_constant(
            ix_start_ep, ix_end_ep, ext, window_size=1, const=np.nan
        )

        # grab the next initial condition, increase the epoch counter
        nan_func = lambda ep_num, x: (ep_num + 1, x[..., ep_num + 1])
    else:
        # single epoch, no augmentation
        nan_func = lambda ep_num, x: (ep_num + 1, x[..., 0])
        aug_data = ext
        idx_start_shift = ix_start_ep
        idx_end_shift = ix_end_ep
        ix_shift = slice(None)


    # call forward recursion
    out = _vmap_recursion_sos(aug_data, sos, z0, nan_func)

    # reverse time axis
    irev = _revert_epochs(idx_start_shift, idx_end_shift)
    out = out.at[ix_shift].set(out[irev])

    # compute new init cond
    z0 = zi[..., jnp.newaxis, jnp.newaxis] * out.T[jnp.newaxis, jnp.newaxis, ..., idx_start_shift]

    # call backward recursion
    out = _vmap_recursion_sos(out, sos, z0, nan_func)

    # re-flip axis
    out = out.at[ix_shift].set(out[irev])

    # remove nans and padding
    out = out[ix_shift][ix_data]

    return out.reshape(original_shape)
