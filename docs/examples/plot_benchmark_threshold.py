"""
# threshold
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
def get_mean_perf_class(obj):
    n = 10
    tmp = np.zeros(n)
    for i in range(n):
        t1 = perf_counter()
        out = obj.threshold(0.0)
        t2 = perf_counter()
        tmp[i] = t2 - t1
    return np.mean(tmp), np.std(tmp)

def benchmark_threshold():
    times = []
    for T in np.arange(500000, 1000000, 100000):
        time_array = np.arange(T) / 2
        data_array = np.random.randn(len(time_array))
        starts = np.arange(1, T // 2 - 1, 20)
        ends = np.arange(1, T // 2 - 1, 20) + 8
        
        ep = nap.IntervalSet(start=starts, end=ends)
        tsd = nap.Tsd(t=time_array, d=data_array, time_support=ep)
        
        res = tsd.threshold(0.0) # First call to compile
        m, s = get_mean_perf_class(tsd)
        times.append([T, m, s])
    return np.array(times)

# %% 
# Calling with numba
nap.nap_config.set_backend("numba")
num_times = benchmark_threshold()


# %%
# Calling with jax

nap.nap_config.set_backend("jax")
jax_times = benchmark_threshold()


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
plt.title("Threshold benchmark")
plt.show()
