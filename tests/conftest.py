import numpy as np
import pytest
import pynapple as nap
from jax import config
from scipy.signal import lfilter

config.update("jax_enable_x64", True)
nap.nap_config.set_backend("jax")
