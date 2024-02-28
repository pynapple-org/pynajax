import itertools

import jax.numpy as jnp
import numpy as np
import pynapple as nap
import pytest

import pynajax as jnap
import pynajax.jax_core
from contextlib import nullcontext as does_not_raise

nap.nap_config.set_backend("jax")

@pytest.mark.parametrize(
    "shape_array1, shape_array2", [((3,), (10, 2)), ((3,), (10, 2, 3))]
)
def test_2d_convolve_epoch_vec(shape_array1, shape_array2):
    """Compare convolution with numpy for 1D kernel"""
    np.random.seed(111)
    arr1 = np.random.normal(size=shape_array1)
    arr2 = np.random.normal(size=shape_array2)

    res_numpy = np.zeros(arr2.shape)
    for indices in itertools.product(*[range(dim) for dim in arr2.shape[1:]]):
        full_indices = (slice(None),) + indices
        res_numpy[full_indices] = np.convolve(arr1, arr2[full_indices], mode="same")

    res_pynajax = jnap.jax_core.convolve_epoch(jnp.asarray(arr2), arr1)
    assert np.allclose(res_pynajax, res_numpy)


@pytest.mark.parametrize(
    "shape_array1, shape_array2", [((3, 2), (10, 2)), ((3, 2), (10, 2, 3))]
)
def test_2d_convolve_epoch_mat(shape_array1, shape_array2):
    """Compare convolution with numpy for 2D kernel"""
    np.random.seed(111)
    arr1 = np.random.normal(size=shape_array1)
    arr2 = np.random.normal(size=shape_array2)

    res_numpy = np.zeros((*arr2.shape, arr1.shape[1]))
    for j in range(arr1.shape[1]):
        for indices in itertools.product(*[range(dim) for dim in arr2.shape[1:]]):
            full_indices = (slice(None),) + indices
            res_numpy[(*full_indices, j)] = np.convolve(
                arr1[:, j], arr2[full_indices], mode="same"
            )

    res_pynajax = jnap.jax_core.convolve_epoch(arr2, arr1)
    assert np.allclose(res_pynajax, res_numpy)
    assert isinstance(res_pynajax, jnp.ndarray)


@pytest.mark.parametrize(
    "iset",
    [
        nap.IntervalSet(start=[0], end=[100]),
        nap.IntervalSet(start=[0, 20], end=[19, 100]),
    ],
)
@pytest.mark.parametrize(
    "data",
    [jnp.ones((100,)), jnp.ones((100, 1)), jnp.ones((100, 2)), jnp.ones((100, 2, 3))],
)
@pytest.mark.parametrize(
    "time",
    [
        np.arange(100),
    ],
)
@pytest.mark.parametrize(
    "kernel",
    [
        np.ones((10,)),
        np.ones((10, 1)),
        np.ones((10, 2)),
        jnp.ones((10,)),
        jnp.ones((10, 1)),
        jnp.ones((10, 2)),
    ],
)
def test_convolve_intervals(time, data, iset, kernel):
    """Run convolution on single and multi interval."""
    nap_data = pynajax.jax_core.construct_nap(time, data, iset, None)
    pynajax.jax_core.convolve_intervals(nap_data, kernel)


@pytest.mark.parametrize(
    "iset",
    [
        nap.IntervalSet(start=[0], end=[100]),
        nap.IntervalSet(start=[0, 20], end=[19, 100]),
    ],
)
@pytest.mark.parametrize(
    "data",
    [jnp.ones((100,)), jnp.ones((100, 1)), jnp.ones((100, 2)), jnp.ones((100, 2, 3))],
)
@pytest.mark.parametrize(
    "time",
    [
        np.arange(100),
    ],
)
@pytest.mark.parametrize(
    "kernel",
    [
        np.ones((10,)),
    ],
)
def test_convolve_intervals_shape_1d_kernel(time, data, iset, kernel):
    """Check that the shape of input and output matches if kernel is 1D."""
    nap_data = pynajax.jax_core.construct_nap(time, data, iset, None)
    expected_shape = nap_data.shape
    res = pynajax.jax_core.convolve_intervals(nap_data, kernel)
    assert all(res.shape[i] == expected_shape[i] for i in range(nap_data.ndim))


@pytest.mark.parametrize(
    "iset",
    [
        nap.IntervalSet(start=[0], end=[100]),
        nap.IntervalSet(start=[0, 20], end=[19, 100]),
    ],
)
@pytest.mark.parametrize(
    "data",
    [jnp.ones((100,)), jnp.ones((100, 1)), jnp.ones((100, 2)), jnp.ones((100, 2, 3))],
)
@pytest.mark.parametrize(
    "time",
    [
        np.arange(100),
    ],
)
@pytest.mark.parametrize(
    "kernel",
    [
        np.ones((10, 1)),
        np.ones((10, 2)),
    ],
)
def test_convolve_intervals_shape_2d_kernel(time, data, iset, kernel):
    """
    Check that the shape of output is equal to that of the input
    plus the second dimension of the kernel attached at the end.
    """
    nap_data = pynajax.jax_core.construct_nap(time, data, iset, None)
    expected_shape = (*nap_data.shape, kernel.shape[1])
    res = pynajax.jax_core.convolve_intervals(nap_data, kernel)
    assert all(res.shape[i] == expected_shape[i] for i in range(nap_data.ndim))

@pytest.mark.parametrize(
    "iset",
    [
        nap.IntervalSet(start=[0], end=[100]),
        nap.IntervalSet(start=[0, 20], end=[19, 100]),
    ],
)
@pytest.mark.parametrize(
    "data",
    [jnp.ones((100,)), jnp.ones((100, 1)), jnp.ones((100, 2)), jnp.ones((100, 2, 3))],
)
@pytest.mark.parametrize(
    "time",
    [
        np.arange(100),
    ],
)
@pytest.mark.parametrize(
    "kernel",
    [
        np.ones((10,)),
        np.ones((10, 1)),
        np.ones((10, 2)),
    ],
)
def test_convolve_construct_nap_type(time, data, iset, kernel):
    """
    Check that the shape of output is equal to that of the input
    plus the second dimension of the kernel attached at the end.
    """
    nap_data = pynajax.jax_core.construct_nap(time, data, iset, None)
    assert isinstance(nap_data.d, jnp.ndarray)


@pytest.mark.parametrize(
    "iset",
    [
        nap.IntervalSet(start=[0], end=[100]),
        nap.IntervalSet(start=[0, 20], end=[19, 100]),
    ],
)
@pytest.mark.parametrize(
    "data",
    [jnp.ones((100,)), jnp.ones((100, 1)), jnp.ones((100, 2)), jnp.ones((100, 2, 3))],
)
@pytest.mark.parametrize(
    "time",
    [
        np.arange(100),
    ],
)
def test_construct_nap_type(time, data, iset):
    """
    Check that the shape of output is equal to that of the input
    plus the second dimension of the kernel attached at the end.
    """
    nap_data = pynajax.jax_core.construct_nap(time, data, iset, None)
    assert isinstance(nap_data.d, jnp.ndarray)


@pytest.mark.parametrize(
    "iset",
    [
        nap.IntervalSet(start=[0], end=[100]),
        nap.IntervalSet(start=[0, 20], end=[19, 100]),
    ],
)
@pytest.mark.parametrize(
    "data, columns",
    [
        (jnp.ones((100, 1)), ["a"]),
        (jnp.ones((100, 2)), ["a", "b"]),
        (jnp.ones((100, 2)), None),
    ],
)
@pytest.mark.parametrize(
    "time",
    [
        np.arange(100),
    ],
)
def test_construct_nap_columns(time, data, iset, columns):
    """
    Check that construct nap defines the proper columns for TsdFrame.
    """
    nap_data = pynajax.jax_core.construct_nap(time, data, iset, columns)
    if columns is None:
        columns = range(data.shape[1])
    assert set(nap_data.columns) == set(columns)
