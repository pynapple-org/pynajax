import pytest
import pynapple as nap
from jax import config

config.update("jax_enable_x64", True)
nap.nap_config.set_backend("jax")

# def pytest_sessionstart(session):
#     """
#     Allows plugins and conftest files to perform initial configuration.
#     This hook is called for every plugin and initial conftest
#     file after command line options have been parsed.
#     """    
#     import pynapple as nap
#     from jax import config
#     config.update("jax_enable_x64", True)
#     nap.nap_config.set_backend("jax")
#     return



