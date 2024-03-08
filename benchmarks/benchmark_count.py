import pynapple as nap
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from time import perf_counter
from pynajax.jax_core_convolve import convolve_epoch
from matplotlib.pyplot import *

jax_times = []
numba_times = []

def get_mean_perf_class(obj, ep):
    n = 10
    tmp = np.zeros(n)
    for i in range(n):
        t1 = perf_counter()
        out = obj.count(1, ep)
        t2 = perf_counter()
        tmp[i] = t2 - t1
    return np.mean(tmp), np.std(tmp)


for nd in range(2, 10):
    print("Dimensions ", nd)

    ep = nap.IntervalSet(start=np.linspace(0, 10000, nd)[0:-1],
        end = np.linspace(0, 10000, nd)[1:]-1)

    nap.config.nap_config.set_backend("jax")

    tsd_jax = nap.Tsd(t=np.arange(10000), d=np.random.randn(10000))
    
    tsd2 = tsd_jax.count(1, ep)

    m, s = get_mean_perf_class(tsd_jax, ep)

    jax_times.append([nd, m, s])
   
    ##############################
    nap.config.nap_config.set_backend("numba")

    tsd = nap.Tsd(t=np.arange(10000), d=np.random.randn(10000))
    
    tsd2 = tsd.count(1, ep)

    m, s = get_mean_perf_class(tsd, ep)

    numba_times.append([nd, m, s])


jax_times = np.array(jax_times)
numba_times = np.array(numba_times)

figure()
for arr, label in zip([numba_times, jax_times], ['pynapple count', 'pynajax count']):
    plot(arr[:,0], arr[:,1], 'o-', label = label)
    fill_between(arr[:,0], arr[:,1] - arr[:,2], arr[:,1] + arr[:,2], alpha = 0.2)
legend()
xlabel("Number of Dimensions")
ylabel("Time (s)")
title("Convolve benchmark")
# savefig("")
show()

