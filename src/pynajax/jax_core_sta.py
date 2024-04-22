import pynapple as nap
import jax
import jax.numpy as jnp
import numpy as np
from typing import Union
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

    filled_d = np.full((time_series.t.shape[0], *data.shape[1:]), out_of_range, dtype=data.dtype)
    fill_idx = 0
    for start, end in zip(starts, ends):
        data_ep = data.get(start, end)
        ts_ep = time_series.get(start, end)
        idxs = np.searchsorted(data_ep.t, ts_ep.t, side="right") - 1
        filled_d[fill_idx:fill_idx + ts_ep.t.shape[0]][idxs >= 0] = data_ep.d[idxs[idxs >= 0]]
        fill_idx += ts_ep.t.shape[0]
    return type(data)(t=time_series.t, d=filled_d, time_support=ep)

def jitperievent_trigger_average(
    time_array,
    count_array,
    time_target_array,
    data_target_array,
    starts,
    ends,
    windows,
    binsize,
):
    T = time_array.shape[0]
    N = count_array.shape[1]
    N_epochs = len(starts)

    time_target_array, data_target_array, count = nap.core._jitted_functions.jitrestrict_with_count(
        time_target_array, data_target_array, starts, ends
    )
    max_count = np.cumsum(count)

    new_data_array = np.full(
        (int(windows.sum()) + 1, count_array.shape[1], *data_target_array.shape[1:]),
        0.0,
    )

    t = 0  # count events

    hankel_array = np.zeros((new_data_array.shape[0], *data_target_array.shape[1:]))

    for k in range(N_epochs):
        if count[k] > 0:
            t_start = t
            maxi = max_count[k]  # epoch start index
            i = maxi - count[k]  # index relative to epoch

            while t < T:
                lbound = time_array[t]
                rbound = np.round(lbound + binsize, 9)

                if time_target_array[i] < rbound:  # compute the start and stop index in the window
                    i_start = i
                    i_stop = i

                    while i_stop < maxi:
                        if time_target_array[i_stop] < rbound:
                            i_stop += 1
                        else:
                            break

                    while i_start < i_stop - 1:
                        if time_target_array[i_start] < lbound:
                            i_start += 1
                        else:
                            break
                    v = np.sum(data_target_array[i_start:i_stop], 0) / float(
                        i_stop - i_start
                    )

                    checknan = np.sum(v)
                    if not np.isnan(checknan):
                        hankel_array[-1] = v

                if t - t_start >= windows[1]:
                    for n in range(N):  # loop over neuron
                        new_data_array[:, n] += (
                            hankel_array * count_array[t - windows[1], n]
                        )

                # hankel_array = np.roll(hankel_array, -1, axis=0)
                hankel_array[0:-1] = hankel_array[1:]
                hankel_array[-1] = 0.0

                t += 1

                i = i_start

                if t == T or time_array[t] > ends[k]:
                    if t - t_start > windows[1]:
                        for j in range(windows[1]):
                            for n in range(N):
                                new_data_array[:, n] += (
                                    hankel_array * count_array[t - windows[1] + j, n]
                                )

                            # hankel_array = np.roll(hankel_array, -1, axis=0)
                            hankel_array[0:-1] = hankel_array[1:]
                            hankel_array[-1] = 0.0

                    hankel_array *= 0.0
                    break

    total = np.sum(count_array, 0)
    for n in range(N):
        if total[n] > 0.0:
            new_data_array[:, n] /= total[n]

    return new_data_array


def pad_and_roll(count_array, window):
    n_samples = count_array.shape[0]
    pad = lambda x: jnp.pad(x, pad_width=((0, window), (0, 0)), constant_values=[0, np.nan])
    roll = jax.vmap(lambda i: jnp.roll(pad(count_array), -i, axis=0))
    return roll(jnp.arange(0, window + 1))[:, :n_samples]


# dot prod shifted counts vs 1D var y vmapped over shift
# [(n_shift, T), (T, )] -> (n_shift, )
dot_prod = jax.vmap(lambda x, y: jnp.nanmean(x*y), in_axes=(0, None), out_axes=0)
# vmap over the neurons
# [(n_shift, T, n_neurons), (T, )] -> (n_shift, n_neurons)
dot_prod_neu = jax.vmap(dot_prod, in_axes=(2, None), out_axes=1)
# vmap over the features
# [(n_shift, T, n_neurons), (T, n_features)] -> (n_shift, n_neurons, n_features)
dot_prod_feature = jax.vmap(dot_prod_neu, in_axes=(None, 1), out_axes=2)


def _batched_dot_prod_feature(data_array, count_array, batch_size, window, shape):
    num_full_batches = data_array.shape[1] // batch_size
    carry = 0

    def scan_fn(carry, x):
        slc = jax.lax.dynamic_slice(data_array, (0, carry), (data_array.shape[0], batch_size))
        batch_result = dot_prod_feature(count_array, slc)
        return carry + batch_size, batch_result

    _, res = jax.lax.scan(scan_fn, carry, None, length=num_full_batches)

    res = jnp.transpose(res, (1, 2, 0, 3))  # move features at the end
    res = res.reshape(*res.shape[:-2],-1) # flatten features

    extra_elements = data_array.shape[1] % batch_size
    if extra_elements:
        # compute residual slice
        resid = dot_prod_feature(count_array, data_array[:, -extra_elements:])
        resid = resid.transpose(1, 2, 0).reshape(*res.shape[:-1], -1)
        res = np.concatenate([res, resid], axis=2)

    # reshape back to original and return
    return res.reshape((window + 1, count_array.shape[2], *shape[1:]))


def _dot_prod_feature_and_reshape(data_array, count_array, window, shape):
    res = dot_prod_feature(count_array, data_array)
    res.reshape((window + 1, count_array.shape[2], *shape[1:]))
    return res


def sta_single_epoch(count_array, data_array, window, batch_size=256):
    count_array = jnp.asarray(count_array, dtype=float)
    data_array = jnp.asarray(data_array, dtype=float)
    shape = data_array.shape
    data_array = data_array.reshape(data_array.shape[0], -1)
    rolled_counts = pad_and_roll(count_array, window)
    if batch_size >= data_array.shape[1]:
        res = _dot_prod_feature_and_reshape(data_array, rolled_counts, window, shape)
    else:
        res = _batched_dot_prod_feature(data_array, rolled_counts, batch_size, window, shape)
    return res


def sta_multi_epoch(time_array, starts, ends, count_array, data_array, window, batch_size=256):
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
        results = sta_single_epoch(cnt, dat, window, batch_size=batch_size)
        return results

    res = sum(
        jnp.divide(
            jax.tree_map(sta, tree_count, tree_data),
            (idx_end - idx_start) / jnp.sum(idx_end - idx_start)
        )
    )

    return res
