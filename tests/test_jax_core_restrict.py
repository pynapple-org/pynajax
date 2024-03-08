import itertools

import jax.numpy as jnp
import numpy as np
import pynapple as nap
import pytest

import pynajax as jnap
from contextlib import nullcontext as does_not_raise

nap.config.nap_config.set_backend("jax")

tsd = nap.Tsd(t=np.arange(100), d=np.random.randn(100))
ep = nap.IntervalSet(start=np.arange(0, 100, 20), end = np.arange(0, 100, 20)+5)

time_array = tsd.t
data_array = tsd.d
starts = ep.start
ends = ep.end

# jnap.restrict(time_array, data_array, starts, ends)


tsd2 = tsd.restrict(ep)



