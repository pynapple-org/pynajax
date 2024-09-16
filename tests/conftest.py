import pynapple as nap
from jax import config

config.update("jax_enable_x64", True)
nap.nap_config.set_backend("jax")
