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
dev_times = []

def get_mean_perf(func, args):
    n = 10
    tmp = np.zeros(n)
    for i in range(n):
        t1 = perf_counter()
        out = func(*args)
        t2 = perf_counter()
        tmp[i] = t2 - t1
    return np.mean(tmp), np.std(tmp)

def get_mean_perf_class(obj, kernel):
    n = 10
    tmp = np.zeros(n)
    for i in range(n):
        t1 = perf_counter()
        out = obj.convolve(kernel)
        t2 = perf_counter()
        tmp[i] = t2 - t1
    return np.mean(tmp), np.std(tmp)


for nd in range(10, 1000, 50):
    print("Dimensions ", nd)
    t = np.arange(10000)
    d = jnp.asarray(np.random.randn(10000, nd))

    nap.nap_config.set_backend("jax")

    tsd_jax = nap.TsdFrame(t=t, d=d)
    jkernel = jnp.ones(11)

    tsd2 = tsd_jax.convolve(jkernel)
    m, s = get_mean_perf_class(tsd_jax, jkernel)

    jax_times.append([nd, m, s])
    
    ##############################
    tsd2 = convolve_epoch(tsd_jax.values, jkernel)
    m, s = get_mean_perf(convolve_epoch, (tsd_jax.values, jkernel))

    dev_times.append([nd, m, s])
   

    ##############################
    nap.nap_config.set_backend("numba")

    tsd = nap.TsdFrame(t=t, d=np.asarray(d))
    
    kernel = np.ones(11)
    
    tsd3 = tsd.convolve(kernel)

    m, s = get_mean_perf_class(tsd, kernel)

    numba_times.append([nd, m, s])


jax_times = np.array(jax_times)
numba_times = np.array(numba_times)
dev_times = np.array(dev_times)

figure()
for arr, label in zip([numba_times, jax_times, dev_times], ['pynapple convolve', 'pynajax convolve', 'Potential pynajax convolve']):
    plot(arr[:,0], arr[:,1], 'o-', label = label)
    fill_between(arr[:,0], arr[:,1] - arr[:,2], arr[:,1] + arr[:,2], alpha = 0.2)
legend()
xlabel("Number of Dimensions")
ylabel("Time (s)")
title("Convolve benchmark")
# savefig("")
show()

