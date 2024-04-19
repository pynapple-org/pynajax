import pynapple as nap
import jax
import jax.numpy as jnp
import numpy as np
from typing import Union

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
        filled_d[fill_idx:fill_idx + ts_ep.t.shape[0]][idxs >= 0] = data_ep.d[idxs[idxs>=0]]
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

def sta_single_epoch(count_array, data_array, window):
    count_array = jnp.asarray(count_array, dtype=float)
    data_array = jnp.asarray(data_array, dtype=float)
    shape = data_array.shape
    data_array = data_array.reshape(data_array.shape[0], -1)
    rolled_counts = pad_and_roll(count_array, window)

    result = dot_prod_feature(rolled_counts, data_array)
    return result.reshape((window + 1, count_array.shape[1], *shape[1:]))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ep = nap.IntervalSet(0, 100)
    feature = nap.Tsd(
        t=np.arange(0, 101, 0.01), d=np.sin(np.arange(int(101 / 0.01))/10), time_support=ep
    )
    t1 = np.arange(1, 100)
    x = np.arange(100, 10000, 100)
    feature[x] = 1.0
    group = nap.TsGroup(
        {0: nap.Ts(t1), 1: nap.Ts(t1 - 0.1), 2: nap.Ts(t1 + 0.2)}, time_support=ep
    )
    binsize=0.2
    time_unit = "s"
    windowsize = (0.6, 0.6)

    ## precomputation
    binsize = nap.TsIndex.format_timestamps(
        np.array([binsize], dtype=np.float64), time_unit
    )[0]
    start = np.abs(
        nap.TsIndex.format_timestamps(
            np.array([windowsize[0]], dtype=np.float64), time_unit
        )[0]
    )
    end = np.abs(
        nap.TsIndex.format_timestamps(
            np.array([windowsize[1]], dtype=np.float64), time_unit
        )[0]
    )

    idx1 = -np.arange(0, start + binsize, binsize)[::-1][:-1]
    idx2 = np.arange(0, end + binsize, binsize)[1:]
    time_idx = np.hstack((idx1, np.zeros(1), idx2))

    eta = np.zeros((time_idx.shape[0], len(group), *feature.shape[1:]))

    windows = np.array([len(idx1), len(idx2)])

    # Bin the spike train
    count = group.count(binsize, ep)

    time_array = np.round(count.index.values - (binsize / 2), 9)
    count_array = count.values
    starts = ep.start
    ends = ep.end

    time_target_array = feature.index.values
    data_target_array = feature.values

    dataset = nap.load_file("/Users/ebalzani/Code/generalized-linear-models/docs/data/m691l1.nwb")

    epochs = dataset["epochs"]
    units = dataset["units"]
    stimulus = dataset["whitenoise"]

    spikes = units[[34]]
    counts = spikes.count(0.01)

    stimulus = fill_forward(counts, stimulus)

    window = 10
    res = sta_single_epoch(counts.d, stimulus.d, window)
    fig, axs = plt.subplots(1, window)
    for k in range(window):
        axs[k].imshow(res[k, 0])
    plt.tight_layout()
    #res = jitperievent_trigger_average(time_array, count_array, time_target_array, np.expand_dims(data_target_array, -1), starts, ends, windows, binsize)

