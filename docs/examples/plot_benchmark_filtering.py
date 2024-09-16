"""
# filtering
"""

import numpy as np
import pynapple as nap
import jax.numpy as jnp
from time import perf_counter
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# %%
# Machine Configuration
import jax
print(jax.devices())


# %%
def get_mean_perf(tsd, mode, n=10):
    tmp = np.zeros(n)
    for i in range(n):
        t1 = perf_counter()
        _ = nap.apply_lowpass_filter(tsd, 0.25 * tsd.rate, mode=mode)
        t2 = perf_counter()
        tmp[i] = t2 - t1
    return [np.mean(tmp), np.std(tmp)]

# %%

def benchmark_time_points(mode):
    times = []
    for T in np.arange(1000, 100000, 20000):
        time_array = np.arange(T)/1000
        data_array = np.random.randn(len(time_array))
        startend = np.linspace(0, time_array[-1], T//100).reshape(T//200, 2)
        ep = nap.IntervalSet(start=startend[::2,0], end=startend[::2,1])
        tsd = nap.Tsd(t=time_array, d=data_array, time_support=ep)
        times.append([T]+get_mean_perf(tsd, mode))
    return np.array(times)

def benchmark_dimensions(mode):
    times = []
    T = 60000
    for n in np.arange(1, 100, 20):
        time_array = np.arange(T)/1000
        data_array = np.random.randn(len(time_array), n)
        startend = np.linspace(0, time_array[-1], T//100).reshape(T//200, 2)
        ep = nap.IntervalSet(start=startend[::2,0], end=startend[::2,1])
        tsd = nap.TsdFrame(t=time_array, d=data_array)#, time_support=ep)
        times.append([n]+get_mean_perf(tsd, mode))
    return np.array(times)


# %%
# # Increasing number of time points
#
# Calling with numba/scipy
nap.nap_config.set_backend("numba")

times_sinc_numba = benchmark_time_points(mode="sinc")
times_butter_scipy = benchmark_time_points(mode="butter")

# %%
# Calling with jax
nap.nap_config.set_backend("jax")

times_sinc_jax = benchmark_time_points(mode="sinc")
times_butter_jax = benchmark_time_points(mode="butter")

# # %%
# Figure


plt.figure(figsize = (16, 5))
plt.subplot(121)
for arr, label in zip(
    [times_sinc_numba, times_sinc_jax],
    ["Sinc (numba)", "Sinc (jax)"],
    ):
    plt.plot(arr[:, 0], arr[:, 1], "o-", label=label)
    plt.fill_between(arr[:, 0], arr[:, 1] - arr[:, 2], arr[:, 1] + arr[:, 2], alpha=0.2)
plt.legend()
plt.xlabel("Number of time points")
plt.ylabel("Time (s)")
plt.title("Windowed-sinc low pass")

plt.subplot(122)
for arr, label in zip(
    [times_butter_scipy, times_butter_jax],
    ["Butter (scipy)", "Butter (jax)"],
    ):
    plt.plot(arr[:, 0], arr[:, 1], "o-", label=label)
    plt.fill_between(arr[:, 0], arr[:, 1] - arr[:, 2], arr[:, 1] + arr[:, 2], alpha=0.2)

plt.legend()
plt.xlabel("Number of time points")
plt.ylabel("Time (s)")
plt.title("Butterworth filter low pass")
# plt.show()


# %%
# # Increasing number of dimensions
#
# Calling with numba/scipy
nap.nap_config.set_backend("numba")

dims_sinc_numba = benchmark_dimensions(mode="sinc")
dims_butter_scipy = benchmark_dimensions(mode="butter")

# %%
# Calling with jax
nap.nap_config.set_backend("jax")

dims_sinc_jax = benchmark_dimensions(mode="sinc")
dims_butter_jax = benchmark_dimensions(mode="butter")

# # %%
# Figure


plt.figure(figsize = (16, 5))
plt.subplot(121)
for arr, label in zip(
    [dims_sinc_numba, dims_sinc_jax],
    ["Sinc (numba)", "Sinc (jax)"],
    ):
    plt.plot(arr[:, 0], arr[:, 1], "o-", label=label)
    plt.fill_between(arr[:, 0], arr[:, 1] - arr[:, 2], arr[:, 1] + arr[:, 2], alpha=0.2)
plt.legend()
plt.xlabel("Number of dimensions")
plt.ylabel("Time (s)")
plt.title("Windowed-sinc low pass")

plt.subplot(122)
for arr, label in zip(
    [dims_butter_scipy, dims_butter_jax],
    ["Butter (scipy)", "Butter (jax)"],
    ):
    plt.plot(arr[:, 0], arr[:, 1], "o-", label=label)
    plt.fill_between(arr[:, 0], arr[:, 1] - arr[:, 2], arr[:, 1] + arr[:, 2], alpha=0.2)

plt.legend()
plt.xlabel("Number of dimensions")
plt.ylabel("Time (s)")
plt.title("Butterworth filter low pass")
plt.show()

