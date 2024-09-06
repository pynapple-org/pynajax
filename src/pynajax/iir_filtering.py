from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .utils import (_get_idxs, _get_shifted_indices, _get_slicing,
                    _odd_ext_multiepoch, _revert_epochs)


@jax.jit
def _recursion_loop_sos(signal, sos, zi):
    """
    Recursion loop for an IIR sos filter.

    Parameters
    ----------
    signal:
        The signal to be filtered.
    sos:
        Second order filter represented as sos.
    zi:
        Initial conditions.

    Returns
    -------
    :
        The filtered signal.

    """

    def internal_loop(s, x_zi):
        x_cur, zi_slice = x_zi
        x_new = sos[s, 0] * x_cur + zi_slice[s, 0]
        zi_slice = zi_slice.at[s, 0].set(
            sos[s, 1] * x_cur - sos[s, 4] * x_new + zi_slice[s, 1]
        )
        zi_slice = zi_slice.at[s, 1].set(sos[s, 2] * x_cur - sos[s, 5] * x_new)
        x_cur = x_new
        return x_cur, zi_slice

    def recursion_step(carry, x):

        x_cur, zi_slice = jax.lax.fori_loop(
            lower=0, upper=sos.shape[0], body_fun=internal_loop, init_val=(x, carry)
        )

        # Use jax.lax.cond to choose between nan_case and not_nan_case
        zi_slice = jax.lax.cond(
            jnp.isnan(x),  # Condition to check
            lambda _: zi,  # Function to call if x is NaN
            lambda _: zi_slice,  # Function to call if x is not NaN
            operand=None,
        )

        return zi_slice, x_cur

    _, res = jax.lax.scan(recursion_step, zi, signal)

    return res


# vectorize the recursion over signals.
_vmap_recursion_sos = jax.vmap(_recursion_loop_sos, in_axes=(1, None, 2), out_axes=1)


def _insert_constant(idx_start, idx_end, data_array, window_size, const=jnp.nan):
    """
    Insert a constant array with `array.shape[0] = window_size` between tsd epochs.

    Parameters
    ----------
    idx_start: jnp.ndarray
        Epoch start indices.
    idx_end: jnp.ndarray
        Epoch end indices.
    window_size: int
        Number of time points of the constant array.
    const: float
        The constant to be inserted.

    Returns
    -------
    data_array:
        An array with the tsd data, with each epoch interleaved  with "window_size" samples with constant
        value.
    ix_orig:
        The indices corresponding to the samples in the original data array.
    ix_shift:
        The shifted indices, after the constant has been interleaved.
     idx_start_shift:
        Index of the epoch start in the interleaved array.
     idx_end_shift:
        Index of the epoch end in the interleaved array.
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


def _expand_initial_condition(data_shape, idx_start, zi, zi_len):
    # set zis with broadcasting tricks
    zi_idx = (
        idx_start[:, jnp.newaxis]
        + jnp.ones((len(idx_start), 1), dtype=int) * jnp.arange(zi_len)[jnp.newaxis]
    ).flatten()

    # assume zi is the output of scipy.signal.lfilter_zi
    zi_big = jnp.zeros(data_shape).at[zi_idx].set(zi)
    return zi_big


def sosfilter(sos, time_array, data_array, starts, ends, zi=None):
    """
    Filter the data for multiple epochs.

    Parameters
    ----------
    sos:
        The 'sos' representation of the filter.
    time_array: NDArray
        The time array.
    data_array: NDArray
        The data array.
    starts: NDArray
        The array containing epoch starts.
    ends: NDArray
        The array containing epoch ends

    Returns
    -------
    out: NDArray
        The filtered output

    """
    orig_shape = data_array.shape
    data_array = data_array.reshape(data_array.shape[0], -1)

    if zi is None:
        x_zi_shape = list(data_array.shape)
        x_zi_shape[0] = 2
        x_zi_shape = tuple([sos.shape[0]] + x_zi_shape)
        zi = jnp.zeros(x_zi_shape)

    if len(starts) == 1:
        return _vmap_recursion_sos(data_array, sos, zi).reshape(orig_shape)

    else:
        agu_data, ix_orig, ix_shift, idx_start_shift, idx_end_shift = _insert_constant(
            *_get_idxs(time_array, starts, ends), data_array, 1, const=np.nan
        )
        out = _vmap_recursion_sos(agu_data, sos, zi)
        out = jnp.zeros(orig_shape).at[ix_orig].set(out[ix_shift]).reshape(orig_shape)
    return out
