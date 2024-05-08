"""
# event_trigger_average
"""

import numpy as np
import pynapple as nap
import jax.numpy as jnp
from time import perf_counter
import matplotlib.pyplot as plt

# %%
# Machine Configuration
import jax
print(jax.devices())

# %% 
def get_mean_perf_class(tsg, tsd, ep, binsize):
    n = 10
    tmp = np.zeros(n)
    for i in range(n):
        t1 = perf_counter()
        out = nap.compute_event_trigger_average(tsg, tsd, binsize, (binsize*10, binsize*10), ep)
        t2 = perf_counter()
        tmp[i] = t2 - t1
    return np.mean(tmp), np.std(tmp)

def benchmark_event_trigger_average(binsize):
    times = []
    for T in np.arange(100000, 1000000, 100000):
        time_array = np.arange(T) / 2
        data_array = np.arange(16 * T).reshape(T, 4, 2, 2)
        starts = np.arange(1, T // 2 - 1, 20)
        ends = np.arange(1, T // 2 - 1, 20) + 8
        ep = nap.IntervalSet(start=starts, end=ends)
        tsd = nap.TsdTensor(t=time_array, d=data_array)
        tsg = nap.TsGroup(
            {i:nap.Ts(time_array[::2]) for i in range(10)}

            )
        sta = nap.compute_event_trigger_average(tsg, tsd, binsize, (binsize*10, binsize*10), ep)

        m, s = get_mean_perf_class(tsg, tsd, ep, binsize)

        times.append([T, m, s])
    return np.array(times)

# %% 
# Calling with numba
binsize = 1.0

nap.nap_config.set_backend("numba")
num_times = benchmark_event_trigger_average(binsize)


# %%
# Calling with jax

nap.nap_config.set_backend("jax")
jax_times = benchmark_event_trigger_average(binsize)


# %%
# Figure
plt.figure()
for arr, label in zip(
    [num_times, jax_times],
    ["numba backend", "pynajax backend"],
    ):
    plt.plot(arr[:, 0], arr[:, 1], "o-", label=label)
    plt.fill_between(arr[:, 0], arr[:, 1] - arr[:, 2], arr[:, 1] + arr[:, 2], alpha=0.2)
plt.legend()
plt.xlabel("Number of Time points")
plt.ylabel("Time (s)")
plt.title("Event_trigger_average benchmark")
plt.show()


# %% 
# Saving
