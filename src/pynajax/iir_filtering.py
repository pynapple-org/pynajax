import jax
import jax.numpy as jnp

from .utils import _get_idxs, _get_shifted_indices, _get_slicing


# Define the IIR recursion
@jax.jit
def _recursion_loop(b_signal, a):
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
    past_out = jnp.zeros((len(a) - 1,))

    def recursion_step(carry, x):
        out_past = carry
        new = (x - jnp.dot(a[1:], out_past)) / a[0]
        new *= ~jnp.isnan(new)
        # Roll out
        out_past = jnp.hstack([new, out_past[:-1]])
        return out_past, new

    _, res = jax.lax.scan(recursion_step, past_out, b_signal)

    return res


# vectorize the recursion over signals.
_vmap_recursion = jax.vmap(_recursion_loop, in_axes=(1, None), out_axes=1)


# vectorize the convolution
def _conv(signal, b):
    return jnp.convolve(signal, b, mode="full")[: len(signal)]


_vmap_conv = jax.vmap(_conv, in_axes=(1, None), out_axes=1)


@jax.jit
def _iir_filter(b, a, signals):
    """
    Vectorized IIR filtering recursion.

    Parameters
    ----------
    b:
        Recursion coefficients input signal.
    a:
        Recursion coefficients output.
    signals: jax.numpy.ndarray
        The signals, shape (n_samples, n_signals).

    Returns
    -------
    :
        Filtered signal.
    """
    b_sig = _vmap_conv(signals, b)
    return _vmap_recursion(b_sig, a)


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
    return data_array, ix_orig, ix_shift


def lfilter(b, a, time_array, data_array, starts, ends):
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
    if len(starts):
        # interleave with 0
        agu_data, ix_orig, ix_shift = _insert_constant(
            time_array, data_array, starts, ends, len(b) - 1, const=0.0
        )
        # convolve
        b_sig = _vmap_conv(agu_data, b)
        # add nans between epochs
        b_sig = jnp.full(b_sig.shape, jnp.nan).at[ix_shift].set(b_sig[ix_shift])
        # run recursion
        out = _vmap_recursion(b_sig, a)
        # remove the extra values
        out = jnp.zeros(data_array.shape).at[ix_orig].set(out[ix_shift])
    else:
        out = _iir_filter(b, a, data_array)
    return out.reshape(orig_shape)
