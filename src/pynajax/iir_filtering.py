import jax
import jax.numpy as jnp
from functools import partial
from .utils import _get_idxs, _get_shifted_indices, _get_slicing, _revert_epochs, _odd_ext_multiepoch
from scipy.signal import lfilter_zi
import numpy as np

# Define the IIR recursion
@jax.jit
def _recursion_loop_ab(b_signal, a, zi):
    """
    Recursion loop for an IIR filter.

    Parameters
    ----------
    b_signal:
        Convolution between the input and the b coefficients of the filter.
    a:
        a coefficient of the recursion.

    Returns
    -------

    """
    past_out = jnp.zeros((len(a) - 1, ))

    def recursion_step(carry, x):
        i, out_past = carry
        new = (x - jnp.dot(a[1:], out_past)) + zi[i]
        new = jnp.nan_to_num(new, nan=0.)
        # Roll out
        out_past = jnp.hstack([new, out_past[:-1]])
        i += 1
        return (i, out_past), new

    _, res = jax.lax.scan(recursion_step, (0, past_out), b_signal)

    return res


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
        zi_slice = zi_slice.at[s, 0].set(sos[s, 1] * x_cur - sos[s, 4] * x_new + zi_slice[s, 1])
        zi_slice = zi_slice.at[s, 1].set(sos[s, 2] * x_cur - sos[s, 5] * x_new)
        x_cur = x_new
        return x_cur, zi_slice

    def recursion_step(carry, x):
        x_cur, zi_slice = jax.lax.fori_loop(lower=0, upper=sos.shape[0], body_fun=internal_loop, init_val=(x, carry))
        return zi_slice, x_cur

    _, res = jax.lax.scan(recursion_step, zi, signal)

    return res


# vectorize the recursion over signals.
_vmap_recursion_ab = jax.vmap(_recursion_loop_ab, in_axes=(1, None, None), out_axes=1)
_vmap_recursion_sos = jax.vmap(_recursion_loop_sos, in_axes=(1, None, None), out_axes=1)


# vectorize the convolution
def _conv(signal, b):
    return jnp.convolve(signal, b, mode="full")[: len(signal)]


_vmap_conv = jax.vmap(_conv, in_axes=(1, None), out_axes=1)


def _insert_constant(idx_start, idx_end, data_array, window_size, const=jnp.nan):
    """
    Insert a constant array with `array.shape[0] = window_size` between tsd epochs.

    Parameters
    ----------
    time_array: NDArray
        The time array.
    data_array: NDArray
        The data array.
    starts: NDArray
        The array containing epoch starts.
    ends: NDArray
        The array containing epoch ends
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
        idx_start, idx_end, window_size + 1
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
            idx_start[:, jnp.newaxis] +
            jnp.ones((len(idx_start), 1), dtype=int) * jnp.arange(zi_len)[jnp.newaxis]
    ).flatten()

    # assume zi is the output of scipy.signal.lfilter_zi
    zi_big = jnp.zeros(data_shape).at[zi_idx].set(zi)
    return zi_big


@partial(jax.jit, static_argnums=(6, 7))
def _iir_filter_multi(b, a, agu_data, zi_big, ix_orig, ix_shift, new_shape, orig_shape):
    # convolve
    b_sig = _vmap_conv(agu_data, b)

    # add nans between epochs
    b_sig = jnp.full(b_sig.shape, jnp.nan).at[ix_shift].set(b_sig[ix_shift])

    # run recursion
    out = _vmap_recursion_ab(b_sig, a, zi_big)

    # remove the extra values and return
    return jnp.zeros(new_shape).at[ix_orig].set(out[ix_shift]).reshape(orig_shape)


@partial(jax.jit, static_argnums=(4, ))
def _iir_filter_single(b, a, data_array, zi, orig_shape):

    # convolve
    b_sig = _vmap_conv(data_array, b)

    # run recursion
    out = _vmap_recursion_ab(b_sig, a, zi)

    # remove the extra values and return
    return out.reshape(orig_shape)


def _lfilter(b, a, data, zi_big, orig_shape, iir_ab_recursion):
    # normalize
    b /= a[0]
    a /= a[0]
    return iir_ab_recursion(b, a, data, zi_big, orig_shape)


def lfilter(b, a, time_array, data_array, starts, ends, zi=None):
    """
    Filter the data for multiple epochs.

    Parameters
    ----------
    b:
        Recursion coefficients input signal.
    a:
        Recursion coefficients output signal.
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
        zi = jnp.zeros_like(len(a) - 1)


    if len(starts) == 1:
        _iir_recursion = _iir_filter_single
        agu_data = data_array
        idx_start_shift = jnp.array([0])

    else:
        agu_data, ix_orig, ix_shift, idx_start_shift, idx_end_shift = _insert_constant(
            *_get_idxs(time_array, starts, ends), data_array, len(b) - 1, const=0.0
        )
        _iir_recursion = lambda x, y, d, z, s: _iir_filter_multi(x, y, d, z, ix_orig, ix_shift, data_array.shape, s)

    zi = jnp.tile(zi, len(idx_start_shift))
    zi_big = _expand_initial_condition(agu_data.shape[0], idx_start_shift, zi, len(a) - 1)
    out = _lfilter(b, a, agu_data, zi_big, orig_shape, _iir_recursion)
    return out



# TODO> remove padding.
def filtfilt(b, a, time_array, data_array, starts, ends):
    orig_shape = data_array.shape

    # pad data, equivalent to scipy.filtfilt default ("pad" method and "odd" padtype).
    pad_num = 3 * max(len(a), len(b))
    data_array = data_array.reshape(data_array.shape[0], -1)
    ext, ix_start_pad, ix_end_pad = _odd_ext_multiepoch(pad_num, time_array, data_array, starts, ends)

    # get the start/end index of each epoch after padding
    ix_start_orig = np.hstack((ix_start_pad[0], ix_start_pad[1:-1]  + pad_num))
    ix_end_orig = np.hstack((ix_start_orig[1:], ix_end_pad[-1]))

    # add zeros between epochs
    agu_data, ix_orig, ix_shift, idx_start_shift, idx_end_shift = _insert_constant(ix_start_orig, ix_end_orig, ext, len(b) - 1, const=0.)

    # get initial delays
    zi = lfilter_zi(b, a)
    zi = zi.reshape([zi.shape[0]] + [1] * (data_array.ndim - 1))

    # compute initial delays (n_starts, len(zi), *data_array[1:]) and reshape to (n_starts * len(zi), *data_array[1:])
    out0 = zi * ext[ix_start_orig, jnp.newaxis]
    out0 = out0.reshape(-1, *out0.shape[2:])
    out0 = _expand_initial_condition(ext.shape, idx_start=ix_start_orig, zi=out0, zi_len=len(zi))

    # add zeros between epochs
    tot_size = ix_shift[-1] - ix_shift[0] + 1
    out0 = (
        jnp.full((tot_size, *out0.shape[1:]), 0.)
        .at[ix_shift]
        .set(out0[ix_orig])
    )

    # run the forward pass
    _iir_recursion = lambda x, y, d, z, s: _iir_filter_multi(x, y, d, z, ix_orig, ix_shift, ext.shape, s)
    out = _lfilter(b, a, agu_data, out0.flatten(), ext.shape, _iir_recursion)


    # invert epoch by epoch
    irev = _revert_epochs(ix_start_orig, ix_end_orig)
    out = out[irev]

    # re-add zeros
    agu_data = (
        jnp.full((tot_size, *out.shape[1:]), 0.)
        .at[ix_shift]
        .set(out[ix_orig])
    )

    # get the new delays
    out0 = zi * out[ix_start_orig, jnp.newaxis]
    out0 = out0.reshape(-1, *out0.shape[2:])
    out0 = _expand_initial_condition(ext.shape, idx_start=ix_start_orig, zi=out0, zi_len=len(zi))
    out0 = (
        jnp.full((tot_size, *out0.shape[1:]), 0.)
        .at[ix_shift]
        .set(out0[ix_orig])
    )

    # backward pass & re-sort
    out = _lfilter(b, a, agu_data, out0.flatten(), ext.shape, _iir_recursion)[irev]
    # remove padding
    # reshape and return
    # save.
    return out