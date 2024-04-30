"""
# convolve
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
def get_mean_perf_class(obj, kernel):
    n = 10
    tmp = np.zeros(n)
    for i in range(n):
        t1 = perf_counter()
        out = obj.convolve(kernel)
        t2 = perf_counter()
        tmp[i] = t2 - t1
    return np.mean(tmp), np.std(tmp)

def benchmark_convolve(kernel):
    times = []
    for nd in range(10, 500, 50):
        print("Dimensions ", nd)
        t = np.arange(10000)
        d = np.random.randn(10000, nd)
        tsd = nap.TsdFrame(t=t, d=d)        
        tsd2 = tsd.convolve(kernel) # First call to compile
        m, s = get_mean_perf_class(tsd, kernel)
        times.append([nd, m, s])
    return np.array(times)

# %%
# Calling with jax
nap.nap_config.set_backend("jax")
jax_times = benchmark_convolve(jnp.ones(11))

# %% 
# Calling with numba
nap.nap_config.set_backend("numba")
num_times = benchmark_convolve(np.ones(11))

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
plt.xlabel("Number of Dimensions")
plt.ylabel("Time (s)")
plt.title("Convolve benchmark")
plt.savefig("../images/convolve_benchmark.png")
plt.show()


# %% 
# Saving

