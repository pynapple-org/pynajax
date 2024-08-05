import pynapple as nap


def test_backend():
    assert nap.nap_config.backend == "jax"
