"""count functions"""

from functools import partial

import jax
import jax.numpy as jnp
import pynapple as nap


@jax.jit
def _hist(time_array, bins):
    return jnp.histogram(time_array, bins=bins)[0]


@jax.jit
def _get_bin_center(bins):
    return bins[0:-1] + (bins[1:] - bins[0:-1]) / 2


# @jax.jit
def count(time_array, starts, ends, bin_size=None):
    """
    Restrict function of pynapple for `Tsd` objects
    """
    if isinstance(bin_size, (float, int)):
        tree = [jnp.arange(s, e + bin_size, bin_size) for s, e in zip(starts, ends)]
    else:
        tree = [jnp.array([s, e]) for s, e in zip(starts, ends)]

    func = partial(_hist, time_array=time_array)

    new_data_array = jnp.concatenate(jax.tree_map(lambda x: func(bins=x), tree), axis=0)

    new_time_array = jnp.concatenate(
        jax.tree_map(lambda x: _get_bin_center(x), tree), axis=0
    )

    return new_time_array, new_data_array


if __name__ == "__main__":
    from time import perf_counter

    import numpy as np

    T = 10000
    time_array = np.sort(np.random.uniform(low=0, high=T, size=T // 10))

    starts = np.arange(1, T - 1, 20)
    ends = np.arange(1, T - 1, 20) + 10

    ep = nap.IntervalSet(start=starts, end=ends)
    ts = nap.Ts(t=time_array)

    count(time_array, starts, ends, bin_size=5)
    t0 = perf_counter()
    count(time_array, starts, ends, bin_size=5)
    print("pynajax count", perf_counter() - t0)

    ts.count(ep, bin_size=5)
    t0 = perf_counter()
    ts.count(ep, bin_size=5)
    print("pynapple count", perf_counter() - t0)
