from functools import partial

import jax
import jax.numpy as jnp


# Define the IIR recursion
@jax.jit
def recursion_loop(b_signal, a):
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
    past_out = jnp.zeros((len(a) - 1, 1))

    first_element = jnp.zeros(past_out.shape)
    first_element = first_element.at[0].set(1)
    other_elements = jnp.ones(past_out.shape) - first_element

    def recursion_step(carry, x):
        out_past = carry
        new = (x - jnp.dot(a[1:], out_past)) / a[0]
        # Roll out
        out_past = first_element * new + other_elements * out_past
        return out_past, new

    _, res = jax.lax.scan(recursion_step,  past_out, b_signal)

    return res


# vectorize the recursion over signals.
vmap_recursion = jax.vmap(recursion_loop, in_axes=(1, None), out_axes=1)

# vectorize the convolution
_conv = jax.vmap(partial(jnp.convolve, mode="same"), in_axes=(1, None),out_axes=1)


@jax.jit
def iir_filter(signals, b, a):
    """
    Vectorized IIR filtering recursion.

    Parameters
    ----------
    signals: jax.numpy.ndarray
        The signals, shape (n_samples, n_signals).
    b:
        Recursion coefficients input signal.
    a:
        Recursion coefficients output.
    Returns
    -------
    :
        Filtered signal.
    """
    b_sig = _conv(signals, b)
    return vmap_recursion(b_sig, a)
