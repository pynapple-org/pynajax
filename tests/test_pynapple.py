import pytest


def test_import_pynapple_and_set_backend():
	import pynapple as nap
	nap.nap_config.set_backend("jax")

	
