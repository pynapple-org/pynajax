import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from .utils import _get_idxs, _get_shifted_indices, _get_slicing, _revert_epochs, _odd_ext_multiepoch
from scipy.signal import lfilter_zi


# Define the IIR recursion
@jax.jit
def _recursion_loop(b_signal, a, zi):
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
        new *= ~jnp.isnan(new)
        # Roll out
        out_past = jnp.hstack([new, out_past[:-1]])
        i += 1
        return (i, out_past), new

    _, res = jax.lax.scan(recursion_step, (0, past_out), b_signal)

    return res


# vectorize the recursion over signals.
_vmap_recursion = jax.vmap(_recursion_loop, in_axes=(1, None, None), out_axes=1)


# vectorize the convolution
def _conv(signal, b):
    return jnp.convolve(signal, b, mode="full")[: len(signal)]


_vmap_conv = jax.vmap(_conv, in_axes=(1, None), out_axes=1)


def _insert_constant(time_array, data_array, starts, ends, window_size, const=jnp.nan):
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
    # get the indexes of start and end
    idx_start, idx_end = _get_idxs(time_array, starts, ends)
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


def _compute_initial_cond(num_pts, idx_start_shift, zi):
    # set zis with broadcasting tricks
    zi_idxs = (
            idx_start_shift[:, jnp.newaxis] +
            jnp.ones((len(idx_start_shift), 1), dtype=int) * jnp.arange(len(zi))[jnp.newaxis]
    ).flatten()

    # set the zis at the start of each epoch
    if zi.ndim == 1:
        zi = jnp.tile(zi, len(idx_start_shift))
    else:
        zi = zi.reshape(-1, )
    zi_big = jnp.zeros(num_pts).at[zi_idxs].set(zi)
    return zi_big


@partial(jax.jit, static_argnums=(6, 7))
def _iir_filter_multi(b, a, agu_data, zi_big, ix_orig, ix_shift, new_shape, orig_shape):
    # convolve
    b_sig = _vmap_conv(agu_data, b)

    # add nans between epochs
    b_sig = jnp.full(b_sig.shape, jnp.nan).at[ix_shift].set(b_sig[ix_shift])

    # run recursion
    out = _vmap_recursion(b_sig, a, zi_big)

    # remove the extra values and return
    return jnp.zeros(new_shape).at[ix_orig].set(out[ix_shift]).reshape(orig_shape)


@partial(jax.jit, static_argnums=(4, ))
def _iir_filter_single(b, a, data_array, zi, orig_shape):

    # set the zis at the start of each epoch
    zi_big = jnp.zeros(len(data_array)).at[:len(zi)].set(zi)

    # convolve
    b_sig = _vmap_conv(data_array, b)

    # run recursion
    out = _vmap_recursion(b_sig, a, zi_big)

    # remove the extra values and return
    return out.reshape(orig_shape)



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
        zi = jnp.zeros((len(a) - 1, ))

    if len(starts) > 1:
        # normalize
        b /= a[0]
        a /= a[0]



        agu_data, ix_orig, ix_shift, idx_start_shift, idx_end_shift = _insert_constant(
            time_array, data_array, starts, ends, len(b) - 1, const=0.0
        )
        # interleave with 0
        zi_big = _compute_initial_cond(len(agu_data), idx_start_shift, zi)
        out = _iir_filter_multi(b, a, agu_data, zi_big, ix_orig, ix_shift, data_array.shape, orig_shape)
    else:
        idx_start = np.searchsorted(time_array, starts[0])
        idx_end = np.searchsorted(time_array, ends[0], side="right")
        out = _iir_filter_single(b, a, data_array[idx_start: idx_end], zi, orig_shape)
    return out


# TODO> return the starts and ends of the padded indices.
# TODO> appropriately create (zi.shape[0], num_epochs, *data.shape[1:]) array to be used as initial condition
# TODO> generalize the z_big creating function, adding the appropriate zis at the right sample.
#       strategy: do it for one signal (this is what we have), vmap to multi-signal, use static args.
# def filtfilt(b, a, time_array, data_array, starts, ends):
#
#     # equivalent to scipy.filtfilt default ("pad" method and "odd" padtype).
#     pad_num = 3 * max(len(a), len(b))
#
#     ext, idx_start_shift, idx_end_shift = _odd_ext_multiepoch(pad_num, time_array, data_array, starts, ends)
#
#     zi = lfilter_zi(b, a)
#     zi_shape = [1] + [1] * data_array.ndim
#     zi_shape[0] = zi.size
#     zi = np.reshape(zi, zi_shape)
#
#     # create z_big: size is ext.shape, has the correct zi per each epoch
#
#     # scorporate the computation of zi_big here.
#
#     out = lfilter(b, a, time_array, data_array, starts, ends, zi=zi*ext[0])
#     irev = _revert_epochs(time_array, starts, ends)
#     out = lfilter(b, a, time_array, out[irev], starts, ends, zi=zi*out[-1])[irev]  # multiply zi by the start out
#     return out